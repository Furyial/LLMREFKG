import os
import pandas as pd
import requests
import json
import re

# 配置API的URL
url_generate = "http://localhost:11434/api/generate"

# 读取Excel文件（路径改为通过环境变量提供，避免暴露真实数据位置）
prompt_excel_path = os.environ.get("PROMPT_EXCEL_PATH")
if not prompt_excel_path:
    raise RuntimeError("请通过环境变量 PROMPT_EXCEL_PATH 提供敏感数据文件路径")
df = pd.read_excel(prompt_excel_path)

# 通过环境变量引入提示词，避免在代码中暴露核心提示内容
def _load_prompt(env_key):
    prompt_value = os.environ.get(env_key)
    if not prompt_value:
        raise RuntimeError(f"缺少敏感提示词：请设置环境变量 {env_key}")
    return prompt_value

rules_and_terms_stage_1 = _load_prompt("NER_STAGE1_PROMPT")
rules_and_terms_stage_2 = _load_prompt("NER_STAGE2_PROMPT")

# 有效实体类型列表
valid_entity_types = ["地点", "人物", "机构"]

# 定义一个函数来获取API的响应
def get_response(url, data):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 检查请求是否成功
        response_dict = response.json()
        return response_dict.get("response", "")
    except Exception as e:
        print(f"Error occurred while fetching response: {e}")
        return ""

# 清理输出
def clean_output(output):
    output = output.strip()  # 去除多余空格
    output = output.replace("“", "\"").replace("”", "\"")  # 替换中文引号
    return output

# 验证并过滤输出
def filter_entities_by_type(raw_output, expected_type):
    filtered_entities = []
    for match in validate_output(raw_output):
        try:
            entity_name, entity_type = json.loads(match)
            if entity_type == expected_type:  # 验证实体类型是否一致
                filtered_entities.append([entity_name, entity_type])
        except json.JSONDecodeError:
            continue
    return filtered_entities

# 验证输出格式
def validate_output(output):
    pattern = r'\[".*?",".*?"\]'  # 匹配 ["实体名称","实体类型"]
    return re.findall(pattern, output)

# 存储最终结果
results = []

# 遍历所有问题并处理
for index, row in df.iterrows():
    prompt = row[0]

    # 第一阶段：识别实体类型
    full_prompt_stage_1 = f"{rules_and_terms_stage_1}\n{prompt}"
    data_stage_1 = {
        "model": "qwen2.5:14b-instruct-q5_K_M",
        "prompt": full_prompt_stage_1,
        "stream": False
    }
    raw_entity_types = get_response(url_generate, data_stage_1)
    raw_entity_types = clean_output(raw_entity_types)
    entity_types_list = [etype.strip() for etype in raw_entity_types.split(",") if etype.strip() in valid_entity_types]

    sentence_results = {'sentence': prompt}

    # 第二阶段：对于每个实体类型，提取相应的实体
    for entity_type in entity_types_list:
        full_prompt_stage_2 = f"{rules_and_terms_stage_2.format(entity_type=entity_type)}\n文本内容: {prompt}"
        data_stage_2 = {
            "model": "qwen2.5:14b-instruct-q5_K_M",
            "prompt": full_prompt_stage_2,
            "stream": False
        }
        raw_entity_output = get_response(url_generate, data_stage_2)
        raw_entity_output = clean_output(raw_entity_output)

        # 验证并过滤实体
        #valid_entities = filter_entities_by_type(raw_entity_output, entity_type)
        #sentence_results[entity_type] = valid_entities if valid_entities else []

    results.append(raw_entity_output)

# 将结果转换为DataFrame，并保存到CSV文件中
df_results = pd.DataFrame(results)
df_results.to_csv('NER_3stage_data.csv', index=False, encoding='utf-8-sig')
