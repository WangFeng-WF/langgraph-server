# -*- coding: utf-8 -*-
"""
从私有库获取报表提示词

工作流程：接收参数→大模型结构化输出→根据参数组装SQL→执行SQL→返回结果
工具：MCP工具，需调用MCP服务器上的selectBySql工具执行SQL查询
大模型：通义千问的qwen-max-latest
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage
from src.tools.dashscope import ChatDashscope
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

class AiPrompt(BaseModel):
    """AI提示词查询参数模型"""
    title: Optional[str] = Field(None, description="指标/提示词标题")
    type: Optional[str] = Field(None, description="指标类型")
    fields: Optional[str] = Field(None, description="业务域")

    def build_where_conditions(self) -> str:
        """根据属性值动态构建WHERE条件"""
        conditions = []

        # 检查title字段
        if self.title and self.title != "*" and self.title.strip():
            conditions.append(f"title = '{self.title}'")

        # 检查type字段
        if self.type and self.type != "*" and self.type.strip():
            conditions.append(f"type = '{self.type}'")

        # 检查fields字段
        if self.fields and self.fields != "*" and self.fields.strip():
            conditions.append(f"fields = '{self.fields}'")

        # 如果没有任何有效条件，默认返回空字符串
        if not conditions:
            return ""

        return " AND ".join(conditions)

try:
    merged_conf = {'api_key': AppConfig.LLM_API_KEY,
                   'base_url': AppConfig.LLM_BASE_URL,
                   'extra_body': {'enable_thinking': False}, 'max_retries': 3, 'model': AppConfig.LLM_MODEL_NAME}
    llm = ChatDashscope(**merged_conf)
    logger.info("通义千问模型 (qwen-max) 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败，请检查API KEY。错误: {e}")
    exit()

async def parse_params_to_structured_output(params_description: str) -> AiPrompt:
    """使用大模型的结构化输出解析参数"""
    try:
        # 构建解析提示词
        parse_prompt = f"""
请解析以下用户输入的查询参数，提取出标题、类型和业务域信息：

用户输入：
{params_description}

注意：
1. 如果某个字段的值是"*"或为空，请设置为null
2. 提取的字段值不要包含【】符号
3. 严格按照用户输入提取，不要添加额外内容
4. 输出格式必须为JSON
        """

        # 使用with_structured_output方法
        structured_llm = llm.with_structured_output(AiPrompt)
        result = structured_llm.invoke([HumanMessage(content=parse_prompt)])

        logger.info(f"Parsed structured output: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing to structured output: {e}")
        # 降级处理：手动解析
        return manual_parse_params(params_description)


def manual_parse_params(params_description: str) -> AiPrompt:
    """手动解析参数的降级方法"""
    title = None
    type_val = None
    fields = None

    lines = params_description.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '【指标/提示词标题】' in line:
            title = line.split('】')[1].strip() if '】' in line else None
            if title == "*":
                title = None
        elif '【指标类型】' in line:
            type_val = line.split('】')[1].strip() if '】' in line else None
            if type_val == "*":
                type_val = None
        elif '【业务域】' in line:
            fields = line.split('】')[1].strip() if '】' in line else None
            if fields == "*":
                fields = None

    return AiPrompt(title=title, type=type_val, fields=fields)


def build_sql_from_ai_prompt(ai_prompt: AiPrompt) -> str:
    """根据AiPrompt对象构建SQL查询语句"""
    # 固定的SELECT字段
    select_fields = """SELECT
    id AS '主键',
    title AS '指标/提示词标题',
    instruction AS '用途说明',
    create_time AS '创建时间',
    type AS '提示词类型',
    fields AS '提示词业务域',
    inputs AS '输入参数',
    sql_example AS 'SQL示例语句',
    deleted AS '是否删除'"""

    # 构建WHERE条件
    where_conditions = ai_prompt.build_where_conditions()

    # 组装完整SQL
    if where_conditions:
        sql = f"{select_fields}\nFROM ai_prompt\nWHERE {where_conditions}\nLIMIT 1000"
    else:
        # 如果没有任何条件，返回所有记录
        sql = f"{select_fields}\nFROM ai_prompt\nLIMIT 1000"

    return sql


async def execute_sql_with_mcp(sql: str) -> Dict[str, Any]:
    """使用MCP工具执行SQL查询"""
    try:
        async with sse_client(url=MCP_SERVER_CONFIG["url"]) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()

                # 调用selectBySql工具
                result = await session.call_tool("selectBySql", arguments={"sql": sql})

                # 结果处理
                return {
                    "success": True,
                    "data": result.content[0].text if result.content else None,
                    "sql_executed": sql,
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Error executing SQL with MCP: {e}")
        return {
            "success": False,
            "error": f"MCP工具调用失败: {str(e)}",
            "error_type": type(e).__name__
        }

async def query_private_prompts(indicator_name: str, indicator_type: str, business_domain: str) -> Dict[str, Any]:
    """
    使用结构化输出的查询函数

    Args:
        params_description: 用户输入的参数描述

    Returns:
        Dict包含查询结果或错误信息
    """
    try:
        ai_prompt = AiPrompt(
            title=indicator_name if indicator_name != "*" else None,
            type=indicator_type if indicator_type != "*" else None,
            fields=business_domain if business_domain != "*" else None
        )

        # 步骤2: 根据结构化数据构建SQL
        sql = build_sql_from_ai_prompt(ai_prompt)

        # 步骤3: 执行SQL查询
        mcp_result = await execute_sql_with_mcp(sql)

        if mcp_result["success"]:
            return {
                "success": True,
                "message": "查询成功",
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "query_result": mcp_result["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "SQL执行失败",
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "error": mcp_result["error"],
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error in structured query: {e}")
        return {
            "success": False,
            "message": "结构化查询过程出错",
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }

async def query_private_prompts_structured(params_description: str) -> Dict[str, Any]:
    """
    使用结构化输出的查询函数

    Args:
        params_description: 用户输入的参数描述

    Returns:
        Dict包含查询结果或错误信息
    """
    try:
        logger.info(f"Starting structured query for params: {params_description}")

        # 步骤1: 使用大模型解析参数到结构化输出
        ai_prompt = await parse_params_to_structured_output(params_description)

        # 步骤2: 根据结构化数据构建SQL
        sql = build_sql_from_ai_prompt(ai_prompt)

        # 步骤3: 执行SQL查询
        mcp_result = await execute_sql_with_mcp(sql)

        if mcp_result["success"]:
            return {
                "success": True,
                "message": "查询成功",
                "params_description": params_description,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "query_result": mcp_result["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "SQL执行失败",
                "params_description": params_description,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "error": mcp_result["error"],
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error in structured query: {e}")
        return {
            "success": False,
            "message": "结构化查询过程出错",
            "params_description": params_description,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def format_query_result(result: Dict[str, Any]) -> str:
    """格式化查询结果为可读的字符串"""
    if not result["success"]:
        return f"查询失败: {result.get('error', '未知错误')}"

    query_data = result["query_result"]

    # 如果结果是字符串，直接返回
    if isinstance(query_data, str):
        return query_data

    # 如果结果是列表或字典，格式化为JSON
    try:
        return json.dumps(query_data, ensure_ascii=False, indent=2)
    except Exception:
        return str(query_data)


# 同步包装器函数，供其他模块调用
def query_private_prompts_sync(params_description: str) -> Dict[str, Any]:
    """同步版本的查询函数"""
    try:
        return asyncio.run(query_private_prompts_structured(params_description))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return {
            "success": False,
            "message": "同步调用失败",
            "params_description": params_description,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def test_query_functionality():
    """测试查询功能"""
    test_cases = [
        "【指标/提示词标题】生产计划达成率（周）\n【指标类型】指标计算-结果查询SQL\n【业务域】生产",
        "【指标/提示词标题】生产计划达成率（周）\n【指标类型】*\n【业务域】生产",
        "【指标/提示词标题】*\n【指标类型】指标计算-结果查询SQL\n【业务域】生产",
        "【指标/提示词标题】生产计划达成率（周）\n【指标类型】*\n【业务域】*"
    ]

    print("开始测试私有库提示词查询功能...")
    print("=" * 60)

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_input}")
        print("-" * 40)

        try:
            result = query_private_prompts_sync(test_input)
            print(f"查询结果: {format_query_result(result)}")
        except Exception as e:
            print(f"测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    test_query_functionality()
