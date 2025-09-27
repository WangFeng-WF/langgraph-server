# -*- coding: utf-8 -*-
import asyncio
import logging
from typing import Dict, Any, Optional, List

from fastapi import FastAPI
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
# from langserve import add_routes
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic_settings import BaseSettings
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
import sys


# 解决控制台中文乱码问题
def setup_encoding():
    try:
        if os.name == 'nt': os.system('chcp 65001 > nul')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.error(f"编码设置失败: {e}")


setup_encoding()


# ==============================================================================
# 1. 配置管理 (Production Best Practice)
# ==============================================================================
class AppSettings:
    """应用配置，从环境变量加载"""
    MCP_SERVER_URL: str = "http://192.168.58.77:8088/sse"
    MCP_TIMEOUT_SECONDS: int = 60

    # 请将您的API Key设置为环境变量 DASHSCOPE_API_KEY
    DASHSCOPE_API_KEY: str = "sk-ca949c46e4904479927923a41562d4d3"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME: str = "qwen-max-latest"


settings = AppSettings()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. 初始化 LLM 和 Pydantic 模型
# ==============================================================================
try:
    if settings.DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY":
        raise ValueError("请在环境变量或.env文件中设置您的 DASHSCOPE_API_KEY")

    llm = ChatOpenAI(
        openai_api_key=settings.DASHSCOPE_API_KEY,
        openai_api_base=settings.LLM_BASE_URL,
        model_name=settings.LLM_MODEL_NAME,
        max_retries=3,
        streaming=True,
    )
    logger.info(f"大模型 '{settings.LLM_MODEL_NAME}' 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败。错误: {e}")
    exit()


# 增强的Pydantic模型，包含LLM的分析过程
class EnhancedTableRecommendation(BaseModel):
    """表推荐结果模型，包含意图分析"""
    inferred_domain: Optional[str] = Field(None, description="模型从用户输入中推断出的业务领域")
    key_concepts: List[str] = Field(default_factory=list, description="模型从用户输入中提取出的核心概念或实体")
    recommended_tables: List[str] = Field(default_factory=list, description="推荐的表名列表")
    reason: Optional[str] = Field(None, description="综合推荐理由")


# ==============================================================================
# 3. 定义Agent的状态 (State)
# ==============================================================================
class AgentState(TypedDict):
    """定义了图在每一步之间传递的数据结构"""
    prompt_info: str  # 初始用户输入
    all_table_comments: Optional[str] = None
    recommendation_analysis: Optional[EnhancedTableRecommendation] = None  # 推荐分析结果
    table_columns: Optional[str] = None
    final_result: Optional[str] = None
    error_message: Optional[str] = None


# ==============================================================================
# 4. 将外部调用封装成工具 (Tools)
# ==============================================================================
async def _mcp_session_manager(func, tool_name, arguments):
    """辅助函数，管理MCP会话的开启和关闭"""
    try:
        async with sse_client(url=settings.MCP_SERVER_URL) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                return await func(session, tool_name, arguments)
    except Exception as e:
        logger.error(f"MCP会话执行失败: {e}")
        raise


@tool
async def get_all_table_comments_tool() -> str:
    """使用MCP工具获取所有表的简介信息。"""
    logger.info("正在调用工具: get_all_table_comments_tool")

    async def _execute(session, tool_name, arguments):
        result = await session.call_tool(tool_name, arguments)
        content = result.content[0].text if result.content else ""
        logger.info(f"成功获取表简介，长度: {len(content)}")
        return content

    return await _mcp_session_manager(_execute, "getTableComment", {})


@tool
async def get_table_columns_tool(tables: List[str]) -> str:
    """根据提供的表名列表，使用MCP工具获取这些表的字段信息。"""
    if not tables:
        return "错误：表列表为空，无法查询字段信息。"
    logger.info(f"正在调用工具: get_table_columns_tool, 参数: {tables}")

    async def _execute(session, tool_name, arguments):
        result = await session.call_tool(tool_name, arguments)
        content = result.content[0].text if result.content else ""
        logger.info(f"成功获取 {len(tables)} 个表的字段信息。")
        return content

    return await _mcp_session_manager(_execute, "getTableColumn", {"tables": ",".join(tables)})


# ==============================================================================
# 5. 定义图的节点 (Nodes)
# ==============================================================================
async def get_tables_node(state: AgentState) -> Dict[str, Any]:
    """节点1: 获取所有表的简介"""
    logger.info("进入节点: get_tables_node")
    try:
        table_comments = await get_all_table_comments_tool.ainvoke({})
        return {"all_table_comments": table_comments}
    except Exception as e:
        logger.error(f"节点 get_tables_node 失败: {e}")
        return {"error_message": f"获取表简介失败: {e}"}


async def recommend_tables_node(state: AgentState) -> Dict[str, Any]:
    """核心节点: 理解自然语言，分析意图，并推荐表"""
    logger.info("进入核心节点: recommend_tables_node")
    prompt_info = state["prompt_info"]
    table_comments = state["all_table_comments"]

    recommendation_prompt = f"""
你是一位顶级的数据库专家和数据分析师。你的任务是根据用户的自然语言需求和一份数据库表的简介清单，推荐最相关的表。

请遵循以下思考步骤：
1.  **分析需求**: 仔细阅读用户的自然语言需求，识别并提取其中的核心信息，包括：
    *   **业务领域**: 用户在讨论哪个业务范畴？（例如：生产、销售、库存、财务等）
    *   **关键实体与概念**: 提到了哪些关键名词？（例如：订单、产品、客户、物料、设备、工单等）
    *   **用户的目标**: 用户想做什么？（例如：计算“达成率”，查询“数量”，分析“成本”等）

2.  **匹配表**: 将你分析出的业务领域、实体和目标，与下面提供的“数据库表简介清单”进行仔细匹配。寻找表名或表描述中包含相关关键词的表。

3.  **得出结论**: 综合你的分析，推荐最相关的表。并清晰地说明你推断出的业务领域、提取的关键概念以及推荐每一个表的具体理由。

---
用户的自然语言需求:
"{prompt_info}"
---
数据库表简介清单:
{table_comments}
---

请严格按照以下JSON格式返回你的分析和推荐结果。如果无法推荐任何表，请返回一个空的 "recommended_tables" 列表。
"""
    try:
        structured_llm = llm.with_structured_output(EnhancedTableRecommendation, include_raw=False)
        result = await structured_llm.ainvoke(recommendation_prompt)

        if result and result.recommended_tables:
            logger.info(f"表推荐成功，推荐了 {len(result.recommended_tables)} 个表。推断领域: {result.inferred_domain}")
            return {"success": True, "recommendation_analysis": result}
        else:
            logger.info("未能根据用户输入推荐任何表。")
            return {"success": False, "error_message": "未能根据用户输入推荐任何表"}
    except Exception as e:
        logger.error(f"节点 recommend_tables_node 失败: {e}", exc_info=True)
        return {"success": True, "error_message": f"大模型推荐表失败: {e}"}


async def get_columns_node(state: AgentState) -> Dict[str, Any]:
    """节点3: 获取推荐表的字段信息"""
    logger.info("进入节点: get_columns_node")
    recommended_tables = state["recommendation_analysis"].recommended_tables
    try:
        columns_info = await get_table_columns_tool.ainvoke({"tables": recommended_tables})
        return {"table_columns": columns_info}
    except Exception as e:
        logger.error(f"节点 get_columns_node 失败: {e}")
        return {"error_message": f"获取表字段信息失败: {e}"}


def format_output_node(state: AgentState) -> Dict[str, Any]:
    """最终节点: 格式化输出"""
    logger.info("进入节点: format_output_node")
    if state.get("error_message"):
        final_result = f"处理失败: {state['error_message']}"
        return {"final_result": final_result}

    output = ["=" * 60, "智能表推荐分析结果", "=" * 60]

    analysis = state.get("recommendation_analysis")
    if analysis:
        output.append(f"\n>> 用户需求分析:")
        output.append(f"  - 推断业务领域: {analysis.inferred_domain or '未能明确推断'}")
        output.append(
            f"  - 提取关键概念: {', '.join(analysis.key_concepts) if analysis.key_concepts else '未能明确提取'}")

    if analysis and analysis.recommended_tables:
        output.append(f"\n>> 推荐的表:")
        output.append(f"  - 表名: {', '.join(analysis.recommended_tables)}")
        if analysis.reason:
            output.append(f"  - 推荐理由: {analysis.reason}")

        if state.get("table_columns"):
            output.append(f"\n>> 表字段详情:\n" + "-" * 40 + f"\n{state['table_columns']}")
    else:
        output.append("\n>> 结论: 未能找到与您的需求匹配的相关数据表。")

    output.append("\n" + "=" * 60)
    return {"final_result": "\n".join(output)}


# ==============================================================================
# 6. 定义图的边和条件逻辑
# ==============================================================================
def should_get_columns(state: AgentState) -> str:
    """条件边: 判断是否应该获取字段信息"""
    logger.info("判断条件: should_get_columns")
    if state.get("error_message"):
        logger.info("决策: 发现错误，结束流程。")
        return "end"

    analysis = state.get("recommendation_analysis")
    if analysis and analysis.recommended_tables:
        logger.info("决策: 已推荐表，继续获取字段。")
        return "continue"
    else:
        logger.info("决策: 未推荐表，结束流程。")
        return "end"


# ==============================================================================
# 7. 构建并编译图
# ==============================================================================
graph_builder = StateGraph(AgentState)

graph_builder.add_node("get_all_tables", get_tables_node)
graph_builder.add_node("recommend_tables", recommend_tables_node)
graph_builder.add_node("get_columns", get_columns_node)
graph_builder.add_node("format_output", format_output_node)

graph_builder.set_entry_point("get_all_tables")

graph_builder.add_edge("get_all_tables", "recommend_tables")
graph_builder.add_conditional_edges(
    "recommend_tables",
    should_get_columns,
    {"continue": "get_columns", "end": "format_output"}
)
graph_builder.add_edge("get_columns", "format_output")
graph_builder.add_edge("format_output", END)

checkpointer = MemorySaver()
app = graph_builder.compile(checkpointer=checkpointer)

# app = graph_builder.compile()

# ==============================================================================
# 8. 使用 LangServe 部署为 API 服务
# ==============================================================================
api = FastAPI(
    title="Natural Language Table Query Agent API",
    version="2.0",
    description="一个能理解自然语言并推荐数据库表的智能体。",
)


# class AgentInput(BaseModel):
#     prompt_info: str
#
# add_routes(
#     api,
#     app.with_types(input_type=AgentInput, output_type=AgentState),
#     path="/agent",
#     config_keys=["thread_id"],
# )

# ==============================================================================
# 9. 测试 (本地运行)
# ==============================================================================
async def run_test():
    """本地测试函数"""
    test_case = "我想分析一下上周各个产线的生产计划达成率情况，看看实际产量和计划产量之间的差异"
    # test_case = "最近的销售订单有哪些客户，他们都买了什么产品？"
    # test_case = "查数据"

    config = {"configurable": {"thread_id": "test-session-nlp-123"}}

    print(f"--- [Test Case] ---\n{test_case}\n--------------------")

    final_state = await app.ainvoke(
        {"prompt_info": test_case},
        config=config
    )

    print("\n--- [Final Result] ---")
    print(final_state.get('final_result', 'No final result found.'))


if __name__ == "__main__":
    # 取消注释以运行本地测试
    if settings.DASHSCOPE_API_KEY != "YOUR_DASHSCOPE_API_KEY":
        asyncio.run(run_test())
    else:
        print("请先设置您的 DASHSCOPE_API_KEY 再进行测试。")

    # 启动API服务器
    # import uvicorn
    # uvicorn.run(api, host="0.0.0.0", port=8000)
