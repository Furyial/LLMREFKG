import os
from py2neo import Graph, Node, Relationship
import pandas as pd

# 连接到 Neo4j 数据库，敏感配置通过环境变量注入
uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
database_name = os.environ.get("NEO4J_DATABASE")

if not all([username, password, database_name]):
    raise RuntimeError("请设置 NEO4J_USERNAME、NEO4J_PASSWORD 和 NEO4J_DATABASE 环境变量")

graph = Graph(uri, auth=(username, password), name=database_name)

# 清空数据库（可选）
graph.delete_all()

# 读取 Excel 文件（路径同样通过环境变量控制）
triples_file_path = os.environ.get("NEO4J_TRIPLE_EXCEL")
sheet_name = os.environ.get("NEO4J_TRIPLE_SHEET", "Sheet1")

if not triples_file_path:
    raise RuntimeError("请通过环境变量 NEO4J_TRIPLE_EXCEL 指定三元组数据文件路径")

df = pd.read_excel(triples_file_path, sheet_name=sheet_name)

# 导入数据到 Neo4j
for index, row in df.iterrows():
    # 提取三元组信息
    label1 = row["label1"]
    name1 = row["name1"]
    relation = row["relation"]
    name2 = row["name2"]
    label2 = row["label2"]

    # 创建节点
    node1 = Node(label1, name=name1)
    node2 = Node(label2, name=name2)

    # 创建关系
    rel = Relationship(node1, relation, node2)

    # 合并节点和关系到图数据库中
    graph.merge(node1, label1, "name")
    graph.merge(node2, label2, "name")
    graph.merge(rel)

print("数据导入完成！")