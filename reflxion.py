import requests
import pandas as pd
import ast
import json
import os
import re
from tqdm import tqdm
from openai import OpenAI

def call_extraction_model(prompt, api_config):
    """
    使用OpenAI客户端库调用用于关系抽取的大模型 API。
    
    :param prompt: 输入的 prompt
    :param api_config: API配置，包含 {api_key, base_url, model}
    :return: API 返回的生成文本
    """
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=api_config["model"],
        messages=[
            {"role": "system", "content": os.environ.get("REFLXION_EXTRACTION_SYSTEM_PROMPT", "你是一个专业的关系抽取助手。")},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=False
    )
    
    return response.choices[0].message.content

def call_evaluation_model(prompt, api_config):
    """
    使用OpenAI客户端库调用用于评估的大模型 API。
    
    :param prompt: 输入的 prompt
    :param api_config: API配置，包含 {api_key, base_url, model}
    :return: API 返回的生成文本
    """
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=api_config["model"],
        messages=[
            {"role": "system", "content": os.environ.get("REFLXION_EVAL_SYSTEM_PROMPT", "你是一个专业的关系评估助手。")},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=False
    )
    
    return response.choices[0].message.content


def _format_prompt(env_key, **kwargs):
    template = os.environ.get(env_key)
    if not template:
        raise RuntimeError(f"缺少敏感提示词：请设置环境变量 {env_key}")
    return template.format(**kwargs)

def clean_relations_output(output):
    """
    清理模型输出，移除解释性文本，只保留关系三元组
    
    :param output: 模型原始输出
    :return: 清理后的关系三元组列表
    """
    # 查找所有符合 "- 实体A - 关系 - 实体B" 格式的行
    relation_pattern = r"^-\s+(.+?)\s+-\s+(.+?)\s+-\s+(.+?)$"
    
    cleaned_lines = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.startswith("-") and " - " in line:
            # 保留以"-"开头并包含关系三元组格式的行
            cleaned_lines.append(line)
    
    # 如果没有找到任何符合格式的行，返回原始输出
    if not cleaned_lines:
        return output
    
    return "\n".join(cleaned_lines)

def extract_evaluate_reextract(text, entities, extraction_api_config, evaluation_api_config):
    """
    使用Reflxion机制，通过两个不同的大模型实现关系抽取、评估和调整。
    
    :param text: 原始文本
    :param entities: 实体列表
    :param extraction_api_config: 抽取模型的API配置，包含 {api_key, base_url, model}
    :param evaluation_api_config: 评估模型的API配置，包含 {api_key, base_url, model}
    :return: 最终调整后的关系列表和评估报告
    """
    # 指定的关系列表
    target_relations = [
        "又名/古称",
        "流经",
        "发源于",
        "治理/管理",
        "相关人物",
        "用途功能",
        "相关事件",
        "相关物产",
        "相关古镇街巷"
    ]
    
    # Prompt 1: 抽取关系（使用抽取模型）
    prompt1 = _format_prompt(
        "REFLXION_PROMPT_STAGE1",
        target_relations=", ".join(target_relations),
        text=text,
        entities=entities,
    )
    extracted_relations_raw = call_extraction_model(prompt1, extraction_api_config)
    # 清理输出
    extracted_relations = clean_relations_output(extracted_relations_raw)
    # print("初始抽取的关系:\n", extracted_relations)

    # Prompt 2: 评估抽取的关系（使用评估模型）
    prompt2 = _format_prompt(
        "REFLXION_PROMPT_STAGE2",
        target_relations=", ".join(target_relations),
        text=text,
        entities=entities,
        extracted_relations=extracted_relations,
    )
    evaluation_feedback = call_evaluation_model(prompt2, evaluation_api_config)
    # print("评估反馈:\n", evaluation_feedback)

    # Prompt 3: 根据反馈调整并再次抽取（使用抽取模型）
    prompt3 = _format_prompt(
        "REFLXION_PROMPT_STAGE3",
        target_relations=", ".join(target_relations),
        text=text,
        entities=entities,
        extracted_relations=extracted_relations,
        evaluation_feedback=evaluation_feedback,
    )
    final_relations_raw = call_extraction_model(prompt3, extraction_api_config)
    # 清理输出
    final_relations = clean_relations_output(final_relations_raw)
    # print("最终调整后的关系:\n", final_relations)

    return {
        "initial_relations": extracted_relations,
        "evaluation_report": evaluation_feedback,
        "final_relations": final_relations,
        "initial_relations_raw": extracted_relations_raw,
        "final_relations_raw": final_relations_raw
    }

def process_dataset(excel_path, extraction_api_config, evaluation_api_config, output_path=None):
    """
    处理Excel数据集，对每行文本和实体进行关系抽取、评估和调整。
    
    :param excel_path: Excel文件路径
    :param extraction_api_config: 抽取模型的API配置
    :param evaluation_api_config: 评估模型的API配置
    :param output_path: 输出结果的文件路径（默认为None，将在Excel同目录下创建结果文件）
    :return: 处理结果列表
    """
    # 加载数据集
    print(f"正在加载数据集: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # 检查必要的列是否存在
    if 'Text' not in df.columns or 'Entities' not in df.columns:
        raise ValueError("数据集必须包含'Text'和'Entities'列")
    
    results = []
    
    # 遍历每行数据
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
        text = str(row['Text'])
        
        # 处理实体列，尝试解析不同格式
        entities_str = str(row['Entities'])
        try:
            # 尝试作为Python列表字符串解析
            entities = ast.literal_eval(entities_str)
        except (ValueError, SyntaxError):
            # 如果失败，尝试将逗号分隔的字符串分割为列表
            entities = [e.strip() for e in entities_str.split(',')]
        
        # print(f"\n处理第 {idx+1} 行数据:")
        # print(f"文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        # print(f"实体: {entities}")
        
        # 运行关系抽取流程
        try:
            result = extract_evaluate_reextract(text, entities, extraction_api_config, evaluation_api_config)
            result['row_idx'] = idx
            result['text'] = text
            result['entities'] = entities
            results.append(result)
            
            # print(f"完成第 {idx+1} 行处理")
        except Exception as e:
            print(f"处理第 {idx+1} 行时出错: {str(e)}")
            results.append({
                'row_idx': idx,
                'text': text,
                'entities': entities,
                'error': str(e)
            })
    
    # 保存结果
    if output_path is None:
        # 如果未提供输出路径，则在输入文件同目录下创建结果文件
        dir_path = os.path.dirname(excel_path)
        file_name = os.path.splitext(os.path.basename(excel_path))[0]
        output_path = os.path.join(dir_path, f"{file_name}_relation_results12.xlsx")
    
    # 将结果转换为Excel格式并保存
    save_results_to_excel(results, output_path)
    
    print(f"结果已保存至: {output_path}")
    return results

def save_results_to_excel(results, output_path):
    """
    将处理结果保存为Excel文件，只包含原始文本、初始抽取结果和最终抽取结果三列
    
    :param results: 处理结果列表
    :param output_path: 输出Excel文件路径
    """
    # 创建一个简单的数据列表
    data = []
    for result in results:
        # 只取需要的三个列
        data.append({
            'Text': result.get('text', ''),
            'Initial Relations': result.get('initial_relations', ''),
            'Final Relations': result.get('final_relations', '')
        })
    
    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    
    print(f"结果已保存至: {output_path}")

# 示例调用
if __name__ == "__main__":
    # 处理Excel数据集
    excel_path = os.environ.get("RELATION_DATASET_PATH")
    if not excel_path:
        raise RuntimeError("请设置 RELATION_DATASET_PATH 环境变量指向待处理数据集")
    
    # DeepSeek API 的基本配置
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量以访问 DeepSeek 服务")
    
    # 配置两个不同的模型 API
    extraction_model = {
        "api_key": api_key,
        "base_url": base_url,
        "model": os.environ.get("REFLXION_EXTRACTION_MODEL", "deepseek-chat")
    }
    
    evaluation_model = {
        "api_key": api_key,
        "base_url": base_url,
        "model": os.environ.get("REFLXION_EVALUATION_MODEL", "deepseek-reasoner")
    }
    
    # 处理整个数据集
    results = process_dataset(excel_path, extraction_model, evaluation_model)