import os
import pandas as pd
from openai import OpenAI
import pandas as pd
import requests
import json
import re

# 初始化DeepSeek客户端（敏感信息通过环境变量提供）
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not deepseek_api_key:
    raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量以访问受限 API")

client = OpenAI(api_key=deepseek_api_key, base_url=deepseek_base_url)

# 配置API的URL (这里不再需要，因为我们直接用OpenAI SDK)
# url_generate = "http://localhost:11434/api/generate"

# 读取Excel文件（路径通过环境变量提供）
dataset_path = os.environ.get("RELATION_DATASET_PATH")
if not dataset_path:
    raise RuntimeError("请设置 RELATION_DATASET_PATH 环境变量指向数据集文件")
df = pd.read_excel(dataset_path)

# 提示词同样通过环境变量注入以避免泄露
relation_extraction_rules = os.environ.get("RELATION_EXTRACTION_PROMPT")
if not relation_extraction_rules:
    raise RuntimeError("请设置 RELATION_EXTRACTION_PROMPT 环境变量以注入私有提示词")

# 清理输出
def clean_output(output):
    output = output.strip()  # 去除多余空格
    output = output.replace("“", "\"").replace("”", "\"")  # 替换中文引号
    return output

# 解析API响应中的关系
def parse_relations(response):
    try:
        # 1. 将 API 返回的 JSON 字符串解析为 Python 对象
        relations = json.loads(response)

        # 2. 检查解析后的对象是否为列表
        if not isinstance(relations, list):
            print(f"Warning: Expected a list of relations, but got {type(relations)}")
            return []  # 如果不是列表，返回空列表

        # 3. 过滤掉格式不正确的关系
        valid_relations = []
        for relation in relations:
            if isinstance(relation, list) and len(relation) == 5:
                valid_relations.append(relation)
            else:
                print(f"Warning: Skipping invalid relation format: {relation}")

        # 4. 返回有效的关系列表
        return valid_relations
    except json.JSONDecodeError:
        # 如果 JSON 解析失败，返回空列表
        print("Warning: Failed to parse API response as JSON")
        return []

# 安全地解析实体列表
def parse_entities(entity_str):
    if pd.isna(entity_str) or not isinstance(entity_str, str):  # 处理NaN值或非字符串类型
        return []

    # 使用正则表达式匹配实体及其类型
    pattern = r'([^;]+)\s*\(([A-Z]+)\)'  # 匹配实体名称和类型
    matches = re.findall(pattern, entity_str)

    # 将实体列表转换为字符串格式：实体名称 (类型)
    entities = [f"{match[0].strip()} ({match[1]})" for match in matches]
    return entities

# 存储最终结果
results = []

# 遍历所有问题并处理
for index, row in df.iterrows():
    sentence = row.iloc[0]  # 使用 .iloc 访问元素
    provided_entities_str = row.iloc[1]  # 使用 .iloc 访问元素

    provided_entities = parse_entities(provided_entities_str)

    # 构建提示信息，利用提供的实体列表（如果有的话），但不排除API自行识别新的实体
    if provided_entities:
        prompt = f"文本内容: {sentence}\n提供的实体列表: {', '.join(provided_entities)}"
    else:
        prompt = f"文本内容: {sentence}\n注意: 提供的实体列表为空，请从文本中识别所有相关实体。"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": relation_extraction_rules},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    cleaned_relations = clean_output(response.choices[0].message.content)

    results.append({
        'sentence': sentence,
        'provided_entities': ', '.join(provided_entities),  # 将实体列表转换为字符串
        'relations': cleaned_relations
    })

# 将结果转换为DataFrame，并保存到CSV文件中
df_results = pd.DataFrame(results)
df_results.to_csv('Relation_Extraction_deepseeknoexample.csv', index=False, encoding='utf-8-sig')

print("处理完成，结果已保存至 Relation_Extraction_deepseek 文件中。")