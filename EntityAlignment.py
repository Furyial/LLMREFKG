import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import requests
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import math
import geopandas as gpd
import os

# 高德地图 API 密钥、本地模型与数据路径全部通过环境变量注入
AMAP_API_KEY = os.environ.get("AMAP_API_KEY")
LOCAL_BERT_PATH = os.environ.get("LOCAL_BERT_PATH")
if not AMAP_API_KEY or not LOCAL_BERT_PATH:
    raise RuntimeError("请设置 AMAP_API_KEY 与 LOCAL_BERT_PATH 环境变量")

# 加载本地 BERT 模型和 tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH)
    model = BertModel.from_pretrained(LOCAL_BERT_PATH)
except Exception as e:
    print(f"加载BERT模型时出错: {e}")
    raise

# 读取 SHP 文件（河流）
shp_path = os.environ.get("RIVER_SHP_PATH", "")
if os.path.exists(shp_path):
    try:
        rivers_gdf = gpd.read_file(shp_path, encoding='cp936')
        print("成功读取 Shapefile 文件")
    except Exception as e:
        print(f"读取 shapefile 时出错: {e}")
        try:
            # 如果 cp936 失败，尝试使用 latin1
            rivers_gdf = gpd.read_file(shp_path, encoding='latin1')
            print("使用 latin1 编码成功读取 Shapefile 文件")
        except Exception as e:
            print(f"使用备选编码读取 shapefile 时也出错: {e}")
            rivers_gdf = gpd.GeoDataFrame()
else:
    print(f"Shapefile 文件不存在: {shp_path}")
    rivers_gdf = gpd.GeoDataFrame()

# 读取 Excel 文件
try:
    kg1_path = os.environ.get("KG1_EXCEL_PATH")
    kg2_path = os.environ.get("KG2_EXCEL_PATH")

    if not kg1_path or not kg2_path:
        raise RuntimeError("请设置 KG1_EXCEL_PATH 与 KG2_EXCEL_PATH 环境变量")

    kg1_data = pd.read_excel(kg1_path, engine='openpyxl')
    kg2_data = pd.read_excel(kg2_path, engine='openpyxl')
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    raise


# 函数：使用 BERT 计算实体名称的词向量
def get_bert_embedding(text):
    """计算文本的 BERT 词向量"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


# 函数：计算余弦相似度
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0


# 函数：通过高德地图 API 获取经纬度
def get_coordinates(entity_name):
    """根据实体名称获取经纬度"""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={entity_name}&key={AMAP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1' and data['geocodes']:
            location = data['geocodes'][0]['location'].split(',')
            return (float(location[1]), float(location[0]))  # 返回 (纬度, 经度)
    return None


# 函数：计算点到线的最短距离
def point_to_line_distance(point, line):
    """计算点到线的最短距离"""
    nearest = nearest_points(line, point)[0]
    return point.distance(nearest)


# 函数：计算地理相似度（基于高斯内核）
def geographic_similarity(distance, sigma=10):
    """将地理距离转换为相似度"""
    return math.exp(-(distance ** 2) / (2 * (sigma ** 2)))


# 主函数：对齐实体
def align_entities(kg1_data, kg2_data, rivers_gdf, alpha=0.5, threshold=0.8, sigma=10):
    """
    对齐两个知识图谱中的实体
    参数：
    - kg1_data: KG1 的数据 (DataFrame)
    - kg2_data: KG2 的数据 (DataFrame)
    - rivers_gdf: 河流 SHP 数据 (GeoDataFrame)
    - alpha: 语义相似度权重 (0-1)
    - threshold: 对齐阈值
    - sigma: 地理相似度高斯函数参数
    """
    alignments = []
    river_labels = {'干流', '一级河流', '二级河流'}  # 定义河流类型

    for _, row1 in kg1_data.iterrows():
        entity1_name = row1['name1']
        entity1_label = row1['label1']
        entity1_embedding = get_bert_embedding(entity1_name)

        for _, row2 in kg2_data.iterrows():
            entity2_name = row2['name1']
            entity2_label = row2['label1']
            entity2_embedding = get_bert_embedding(entity2_name)

            # 计算语义相似度并归一化到 [0, 1]
            semantic_sim = cosine_similarity(entity1_embedding, entity2_embedding)
            semantic_sim = (semantic_sim + 1) / 2

            # 如果实体类型为 {干流、一级河流、二级河流}，从 SHP 文件中获取线状数据
            geo_sim = 0
            if entity1_label in river_labels:
                river_row = rivers_gdf[rivers_gdf['NAME'] == entity1_name]
                if not river_row.empty:
                    river_geom = river_row.iloc[0]['geometry']  # 获取线状地理数据
                    entity2_coord = get_coordinates(entity2_name)
                    if entity2_coord:
                        point2 = Point(entity2_coord[1], entity2_coord[0])
                        distance = point_to_line_distance(point2, river_geom)
                        geo_sim = geographic_similarity(distance, sigma)

            # 计算综合相似度
            combined_sim = alpha * semantic_sim + (1 - alpha) * geo_sim

            # 判断是否对齐
            if combined_sim >= threshold:
                alignments.append({
                    'KG1_Entity': entity1_name,
                    'KG2_Entity': entity2_name,
                    'Semantic_Similarity': semantic_sim,
                    'Geographic_Similarity': geo_sim,
                    'Combined_Similarity': combined_sim
                })

    return pd.DataFrame(alignments)


# 运行实体对齐
results = align_entities(kg1_data, kg2_data, rivers_gdf, alpha=0.8, threshold=0.8, sigma=10)

# 保存结果到 CSV 文件
results.to_csv('entity_alignments.csv', index=False)
print("对齐结果已保存到 'entity_alignments.csv'")