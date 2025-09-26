# -*- coding: utf-8 -*-
"""
根据AI提示词获取相关的表和字段信息

工作流程：
1. 接收AI提示词参数（支持多个）→大模型结构化输出解析提示词信息
2. 使用MCP工具getTableComment获取所有表简介（参数：{}）
3. 大模型根据提示词信息和表简介，智能匹配可能用到的相关表
4. 调用MCP工具getTableColumn获取匹配表的字段信息（参数：{"tables":"table1,table2"}）
5. 格式化输出表和字段信息

工具：MCP工具，需调用MCP服务器上的getTableComment和getTableColumn工具
大模型：通义千问的qwen-max-latest
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MCP服务器配置
MCP_SERVER_CONFIG = {
    "server_type": "sse",
    "url": "http://192.168.58.77:8088/sse",
    "timeout_seconds": 60
}


class AiPromptAnalysis(BaseModel):
    """AI提示词分析模型"""
    title: Optional[str] = Field(None, description="指标/提示词标题")
    type: Optional[str] = Field(None, description="指标类型")
    fields: Optional[str] = Field(None, description="业务域")
    instruction: Optional[str] = Field(None, description="用途说明")


class MultipleAiPromptAnalysis(BaseModel):
    """多个AI提示词分析模型"""
    prompts: List[AiPromptAnalysis] = Field(default_factory=list, description="多个AI提示词分析结果")


class TableRecommendation(BaseModel):
    """表推荐结果模型"""
    recommended_tables: List[str] = Field(default_factory=list, description="推荐的表名列表")
    reason: Optional[str] = Field(None, description="推荐理由")


def split_multiple_prompts(prompt_info: str) -> List[str]:
    """将多个提示词信息分割为单个提示词列表"""
    # 通过特定分隔符或模式来分割多个提示词
    # 支持多种分割方式：连续的【指标/提示词标题】、空行分割、数字序号等

    # 方式1: 通过【指标/提示词标题】分割
    if prompt_info.count('【指标/提示词标题】') > 1:
        parts = prompt_info.split('【指标/提示词标题】')
        prompts = []
        for i, part in enumerate(parts):
            if i == 0 and not part.strip():
                continue  # 跳过第一个空部分
            if part.strip():
                prompts.append('【指标/提示词标题】' + part.strip())
        return prompts

    # 方式2: 通过连续空行分割
    if '\n\n' in prompt_info:
        parts = prompt_info.split('\n\n')
        return [part.strip() for part in parts if part.strip()]

    # 方式3: 通过数字序号分割（如：1.、2.、3.）
    import re
    pattern = r'\n\s*\d+\.\s*【'
    if re.search(pattern, prompt_info):
        parts = re.split(pattern, prompt_info)
        prompts = []
        for i, part in enumerate(parts):
            if i == 0:
                prompts.append(part.strip())
            else:
                prompts.append('【' + part.strip())
        return [p for p in prompts if p.strip()]

    # 如果无法分割，返回原始内容作为单个提示词
    return [prompt_info.strip()]

import os
from src.tools.dashscope import ChatDashscope

class AppConfig:
    ALLOWED_BUSINESS_DOMAINS = ["生产", "采购", "销售", "供应链"]
    LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ca949c46e4904479927923a41562d4d3")
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME = "qwen-max-latest"

    STEP_MAP = {
        1: "名词定义", 2: "计算公式", 3: "分析维度与指标", 4: "数据来源",
        5: "指标计算-结果查询SQL", 6: "明细计算-实时计算SQL",
        7: "指标分析方法", 8: "明细分析方法"
    }

try:
    merged_conf = {'api_key': AppConfig.LLM_API_KEY,
                   'base_url': AppConfig.LLM_BASE_URL,
                   'extra_body': {'enable_thinking': False}, 'max_retries': 3, 'model': AppConfig.LLM_MODEL_NAME}
    llm = ChatDashscope(**merged_conf)
    logger.info("通义千问模型 (qwen-max) 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败，请检查API KEY。错误: {e}")
    exit()

async def parse_multiple_prompts_to_structured_output(prompt_info: str) -> MultipleAiPromptAnalysis:
    """使用大模型的结构化输出解析多个AI提示词信息"""
    try:
        # 首先尝试分割多个提示词
        # prompt_list = split_multiple_prompts(prompt_info)


            # 多个提示词，批量处理
        parse_prompt = f"""
请仔细分析以下文本，这里包含多个AI提示词信息。请注意：每当出现"【指标/提示词标题】"时，就表示一个新的提示词开始。

请将每个提示词解析为独立的对象，以json格式返回：

提示词信息：
{prompt_info}

解析要求：
1. 识别每个"【指标/提示词标题】"作为新提示词的开始
2. 每个提示词应该包含类似于：【指标/提示词标题】、【指标类型】、【业务域】、【用途说明】，对应的字段分别为：title、type、fields、instruction等字段
3. 提取的字段值不要包含【】符号和冒号
4. 如果某个字段没有明确提到，设置为null
5. 必须返回json格式的结构化数据，包含prompts数组
6. 示例中有{prompt_info.count('【指标/提示词标题】')}个提示词，请解析为{prompt_info.count('【指标/提示词标题】')}个独立对象
"""

        structured_llm = llm.with_structured_output(MultipleAiPromptAnalysis)
        result = structured_llm.invoke([HumanMessage(content=parse_prompt)])

        logger.info(f"Parsed multiple AI prompt analysis: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing multiple prompts to structured output: {e}")
        return manual_parse_multiple_prompts(prompt_info)


async def parse_single_prompt_to_structured_output(prompt_info: str) -> AiPromptAnalysis:
    """使用大模型的结构化输出解析单个AI提示词信息"""
    try:
        parse_prompt = f"""
请解析以下AI提示词信息，提取出所有相关字段：

提示词信息：
{prompt_info}

注意：
1. 提取的字段值不要包含【】符号
2. 严格按照输入内容提取，不要添加额外内容
3. 如果某个字段没有提到，设置为null
4. 特别关注业务域、SQL示例等信息，这些对表推荐很重要
        """

        # llm = get_llm_by_type("basic")
        structured_llm = llm.with_structured_output(AiPromptAnalysis)
        result = structured_llm.invoke([HumanMessage(content=parse_prompt)])

        logger.info(f"Parsed single AI prompt analysis: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing single prompt to structured output: {e}")
        return manual_parse_prompt(prompt_info)


def manual_parse_multiple_prompts(prompt_info: str) -> MultipleAiPromptAnalysis:
    """手动解析多个AI提示词信息的降级方法"""
    prompt_list = split_multiple_prompts(prompt_info)
    analyses = []

    for single_prompt in prompt_list:
        analysis = manual_parse_prompt(single_prompt)
        analyses.append(analysis)

    return MultipleAiPromptAnalysis(prompts=analyses)


def manual_parse_prompt(prompt_info: str) -> AiPromptAnalysis:
    """手动解析AI提示词信息的降级方法"""
    title = None
    type_val = None
    fields = None
    instruction = None

    lines = prompt_info.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '【指标/提示词标题】' in line or '【标题】' in line:
            title = line.split('】')[1].strip() if '】' in line else None
        elif '【指标类型】' in line or '【类型】' in line:
            type_val = line.split('】')[1].strip() if '】' in line else None
        elif '【业务域】' in line:
            fields = line.split('】')[1].strip() if '】' in line else None
        elif '【用法说明】' in line or '【说明】' in line:
            instruction = line.split('】')[1].strip() if '】' in line else None

    return AiPromptAnalysis(
        title=title,
        type=type_val,
        fields=fields,
        instruction=instruction
    )


async def get_all_table_comments() -> Dict[str, Any]:
    """使用MCP工具获取所有表简介"""
    try:
        async with sse_client(url=MCP_SERVER_CONFIG["url"]) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                result = await session.call_tool("getTableComment", arguments={})

                return {
                    "success": True,
                    "data": result.content[0].text if result.content else None,
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Error getting table comments: {e}")
        return {
            "success": False,
            "error": f"获取表简介失败: {str(e)}",
            "error_type": type(e).__name__
        }


async def recommend_tables_by_llm(multiple_analysis: MultipleAiPromptAnalysis, table_comments: str) -> TableRecommendation:
    """使用大模型根据多个AI提示词和表简介推荐相关表"""
    try:
        # 构建包含所有提示词信息的推荐提示
        prompts_summary = []
        for i, analysis in enumerate(multiple_analysis.prompts, 1):
            prompts_summary.append(f"""
提示词{i}：
- 标题：{analysis.title or "无"}
- 类型：{analysis.type or "无"}
- 业务域：{analysis.fields or "无"}
- 用途说明：{analysis.instruction or "无"}
            """)

        recommendation_prompt = f"""
基于以下多个AI提示词信息和数据库表简介，请推荐可能会用到的表。

{''.join(prompts_summary)}

数据库表简介：
{table_comments}

要求：
1. 只推荐数据库表简介中明确存在的表名
2. 表名必须完全匹配数据库表简介中的名称
3. 优先匹配业务域关键词，如"生产"对应包含production、plan、mo等的表名
4. 如果没有合适的表，返回空数组
5. 最多推荐10个表
6. 必须以JSON格式返回结果，包含recommended_tables数组和reason字段

请综合考虑所有提示词的业务域、用途说明等信息进行智能匹配，并以标准JSON格式输出推荐结果。
"""

        # llm = get_llm_by_type("basic")
        structured_llm = llm.with_structured_output(TableRecommendation)
        result = structured_llm.invoke([HumanMessage(content=recommendation_prompt)])

        logger.info(f"Table recommendation result: {result}")

        # 验证推荐结果的有效性
        if result and hasattr(result, 'recommended_tables'):
            return result
        else:
            logger.warning(f"Invalid recommendation result: {result}")
            return TableRecommendation(
                recommended_tables=[],
                reason="结构化输出解析失败，返回空结果"
            )

    except Exception as e:
        logger.error(f"Error in table recommendation: {e}")
        return manual_recommend_tables(multiple_analysis, table_comments)


def manual_recommend_tables(multiple_analysis: MultipleAiPromptAnalysis, table_comments: str) -> TableRecommendation:
    """手动推荐表的降级方法"""
    recommended_tables = []

    # 从table_comments中提取所有可用的表名
    import re

    # 更精确的表名提取模式
    available_tables = []
    lines = table_comments.split('\n')

    for line in lines:
        line = line.strip()
        if line:
            # 匹配各种可能的表名格式
            # 格式1: 表名：描述
            match1 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)[：:](.+)', line)
            # 格式2: 表名 - 描述
            match2 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*(.+)', line)
            # 格式3: 表名 空格 描述
            match3 = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s+(.+)', line)

            if match1:
                available_tables.append(match1.group(1))
            elif match2:
                available_tables.append(match2.group(1))
            elif match3 and len(match3.group(1)) > 3:  # 避免匹配到单词
                available_tables.append(match3.group(1))

    logger.info(f"Extracted available tables: {available_tables}")

    # 业务域关键词映射（更全面）
    business_keywords = {
        "生产": ["production", "plan", "mo", "material", "manufacture", "prod", "work", "order"],
        "销售": ["sale", "order", "customer", "sell", "revenue", "sales"],
        "财务": ["finance", "cost", "price", "financial", "accounting", "budget"],
        "库存": ["inventory", "stock", "warehouse", "store", "material"],
        "质量": ["quality", "qc", "inspection", "check", "defect"],
        "设备": ["equipment", "machine", "device", "facility"],
        "人员": ["employee", "staff", "worker", "personnel", "user"]
    }

    # 处理每个提示词分析结果
    for analysis in multiple_analysis.prompts:
        if analysis.fields:
            for business, keywords in business_keywords.items():
                if business in analysis.fields:
                    # 在可用表中查找匹配的表名
                    for table_name in available_tables:
                        table_lower = table_name.lower()
                        for keyword in keywords:
                            if keyword in table_lower:
                                recommended_tables.append(table_name)
                                break

        # 如果标题中包含关键信息，也进行匹配
        if analysis.title:
            title_lower = analysis.title.lower()
            for table_name in available_tables:
                table_lower = table_name.lower()
                # 简单的标题匹配
                if any(word in table_lower for word in title_lower.split() if len(word) > 2):
                    recommended_tables.append(table_name)

    # 去重并限制数量
    unique_tables = list(dict.fromkeys(recommended_tables))[:10]  # 保持顺序去重

    return TableRecommendation(
        recommended_tables=unique_tables,
        reason=f"基于业务域关键词匹配，从{len(available_tables)}个可用表中推荐{len(unique_tables)}个表"
    )


async def get_table_columns(table_list: List[str]) -> Dict[str, Any]:
    """使用MCP工具获取指定表的字段信息"""
    try:
        if not table_list:
            return {
                "success": False,
                "error": "表列表为空"
            }

        # 将表名列表转换为逗号分隔的字符串
        tables_param = ",".join(table_list)

        async with sse_client(url=MCP_SERVER_CONFIG["url"]) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                result = await session.call_tool("getTableColumn", arguments={"tables": tables_param})

                return {
                    "success": True,
                    "data": result.content[0].text if result.content else None,
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Error getting table columns: {e}")
        return {
            "success": False,
            "error": f"获取表字段信息失败: {str(e)}",
            "error_type": type(e).__name__
        }


async def query_table_info_structured(prompt_info: str) -> Dict[str, Any]:
    """
    使用结构化方式查询表信息的主函数（支持多个提示词）

    Args:
        prompt_info: AI提示词信息（可以是多个）

    Returns:
        Dict包含表和字段信息或错误信息
    """
    try:
        logger.info(f"Starting table info query for prompt: {prompt_info}")

        # 步骤1: 解析AI提示词信息（支持多个）
        multiple_analysis = await parse_multiple_prompts_to_structured_output(prompt_info)

        # 步骤2: 获取所有表简介
        table_comments_result = await get_all_table_comments()
        if not table_comments_result["success"]:
            return {
                "success": False,
                "error": table_comments_result["error"],
                "prompt_info": prompt_info,
                "timestamp": datetime.now().isoformat()
            }

        # 步骤3: 使用大模型推荐相关表
        table_recommendation = await recommend_tables_by_llm(
            multiple_analysis,
            table_comments_result["data"]
        )

        if not table_recommendation.recommended_tables:
            return {
                "success": False,
                "error": "未找到相关的表",
                "prompt_info": prompt_info,
                "parsed_prompts": [p.dict() for p in multiple_analysis.prompts],
                "table_comments": table_comments_result["data"],
                "timestamp": datetime.now().isoformat()
            }

        # 步骤4: 获取推荐表的字段信息
        columns_result = await get_table_columns(table_recommendation.recommended_tables)

        if columns_result["success"]:
            return {
                "success": True,
                "prompt_info": prompt_info,
                "parsed_prompts": [p.dict() for p in multiple_analysis.prompts],
                "recommended_tables": table_recommendation.recommended_tables,
                "recommendation_reason": table_recommendation.reason,
                "table_comments": table_comments_result["data"],
                "table_columns": columns_result["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": columns_result["error"],
                "prompt_info": prompt_info,
                "parsed_prompts": [p.dict() for p in multiple_analysis.prompts],
                "recommended_tables": table_recommendation.recommended_tables,
                "table_comments": table_comments_result["data"],
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error in table info query: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt_info": prompt_info,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def format_table_info_result(result: Dict[str, Any]) -> str:
    """格式化表信息查询结果为可读的字符串"""
    if not result["success"]:
        return f"查询失败: {result.get('error', '未知错误')}"

    output = []
    output.append("=" * 60)
    output.append("表信息查询结果")
    output.append("=" * 60)

    # 解析的提示词信息
    parsed_prompts = result.get("parsed_prompts", [])
    if parsed_prompts:
        output.append(f"\n解析的提示词数量：{len(parsed_prompts)}")
        for i, prompt in enumerate(parsed_prompts, 1):
            output.append(f"\n提示词{i}:")
            output.append(f"  - 标题：{prompt.get('title', '无')}")
            output.append(f"  - 类型：{prompt.get('type', '无')}")
            output.append(f"  - 业务域：{prompt.get('fields', '无')}")

    # 推荐的表
    recommended_tables = result.get("recommended_tables", [])
    if recommended_tables:
        output.append(f"\n推荐的表：{', '.join(recommended_tables)}")
        if result.get("recommendation_reason"):
            output.append(f"推荐理由：{result['recommendation_reason']}")

    # 表简介
    if result.get("table_comments"):
        output.append(f"\n表简介信息：")
        output.append("-" * 40)
        output.append(result["table_comments"])

    # 表字段信息
    if result.get("table_columns"):
        output.append(f"\n表字段信息：")
        output.append("-" * 40)
        output.append(result["table_columns"])

    output.append("\n" + "=" * 60)

    return "\n".join(output)


# 同步包装器函数，供其他模块调用
def query_table_info_sync(prompt_info: str) -> Dict[str, Any]:
    """同步版本的查询表信息函数"""
    try:
        return asyncio.run(query_table_info_structured(prompt_info))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return {
            "success": False,
            "error": str(e),
            "prompt_info": prompt_info,
            "timestamp": datetime.now().isoformat()
        }


def test_table_info_query():
    """测试表信息查询功能（包含多个提示词测试）"""
    test_cases = [
        # 单个提示词测试
        """【指标/提示词标题】：生产计划达成率（周）
【指标类型】：计算公式
【用途说明】：生产计划达成率（周） = ∑实际产量（周） ÷ ∑计划产量（周） × 100%
【业务域】：生产

【指标/提示词标题】：生产计划达成率（周）
【指标类型】：分析维度与指标
【用途说明】：列出常用维度及衍生指标，便于建模与分析。
维度示例：工厂、产线、产品族、订单类型、生产日期、班次。
指标示例：日达成率、周达成率、计划偏差量（实际 - 计划）
【业务域】：生产"""
    ]

    print("开始测试表信息查询功能（支持多提示词）...")
    print("=" * 80)

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:\n{test_input}")
        print("-" * 80)

        try:
            result = query_table_info_sync(test_input)
            print(format_table_info_result(result))
        except Exception as e:
            print(f"测试失败: {str(e)}")

    print("\n" + "=" * 80)
    print("测试完成")


if __name__ == "__main__":
    test_table_info_query()
