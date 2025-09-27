# -*- coding: utf-8 -*-
# manufacturing_agent.py

import os
import sys
import uuid
from typing import Any, Coroutine

from typing_extensions import List, Dict, Optional, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

# ==============================================================================
# 0. 环境和依赖导入
# ==============================================================================
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from src.tools.dashscope import ChatDashscope

from src.tools.promptsQueryFromPrivate import query_private_prompts
from src.tools.promptsQueryFromCloud import query_cloud_prompts
from src.tools.saveOrUpdatePrompts import save_or_update_prompts
import logging
import asyncio

# 在程序开始时配置一次
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台
        # logging.FileHandler("agent.log") # 也可以同时输出到文件
    ]
)

logger = logging.getLogger(__name__)


# 在代码中用 logger 替换 print
# logger.debug("这是一条调试信息")
# logger.info("Agent节点开始处理")
# logger.warning("某个参数可能已弃用")
# logger.error("工具执行失败", exc_info=True) # exc_info=True 会自动附加异常信息

# At the top of the file or in a separate config.py
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


# As a constant
SYSTEM_PROMPT_TEMPLATE_START = "你是「制造业指标分析卡片助手」。你的任务是友好地响应用户，并在用户输入'新增'或'修改'指令时，开始8步引导流程。"

SYSTEM_PROMPT_TEMPLATE_STEP = """你是「制造业指标分析卡片助手」，一个遵循严谨流程的专家。
# 1. 当前上下文
- 指标名称: '{indicator_name}'
- 业务域: '{business_domain}'
- 正在进行的步骤: 第 {step} 步 - {step_name}
- 用户的最新输入: '{user_input}'

# 2. 你的核心任务：为当前步骤生成建议内容
你必须严格遵循以下两步决策流程：

**第一步：查询知识库 (MANDATORY)**
- **行动**: 立即调用 `get_report_prompt` 工具来查询已有的标准内容。
- **参数**:
  - `indicator_name`: '{indicator_name}'
  - `business_domain`: '{business_domain}'
  - `indicator_type`: '{step_name}'
- **目的**: 优先使用经过验证的、标准化的知识。这是你的首要操作，绝不能跳过。

**第二步：决策与响应**
- **如果工具调用成功 (`success: True`)**:
  - **任务**: 完美地采纳工具返回的 `data` 内容。不要修改或添加任何自己的创造。
  - **行动**: 将查询到的内容清晰地呈现给用户，并询问他们是否接受。
  - **示例**: "根据我们的知识库，关于「{indicator_name}」的「{step_name}」，标准内容如下：[此处填充工具返回的data]。请问这个版本是否符合您的要求？"

- **如果工具调用失败 (`success: False`)**:
  - **任务**: 现在轮到你发挥专长。根据你对制造业的理解和行业最佳实践，为「{step_name}」生成一段高质量、专业且具体的建议内容。
  - **行动**: 向用户清晰地说明这是系统生成的建议，并请求他们确认或提出修改意见。
  - **示例**: "我没有在知识库中找到关于「{indicator_name}」的「{step_name}」的标准条目。基于行业最佳实践，我为您生成了以下建议：[此处填充你生成的内容]。您看是否合适？"

# 3. 重要约束
- **不要偏离流程**: 你的唯一任务就是完成上述“查询-决策-响应”的循环。
- **聚焦当前步骤**: 不要讨论与当前「{step_name}」无关的话题。
- **用户输入优先**: 如果用户的最新输入 (`{user_input}`) 已经提供了明确的内容（例如“用这个定义：XXX”），你可以直接使用用户提供的内容，并调用 `save_or_update_prompt` 工具进行保存，然后推进到下一步。
"""

# 【新增】专门为第四步设计的提示词模板
SYSTEM_PROMPT_TEMPLATE_STEP_4 = """你是「制造业指标分析卡片创建助手」，一个拥有深厚制造业知识、数据分析能力和商业智能（BI）实践经验的专家。
# 1. 指标卡片配置流程
你当前处于整个流程最关键的一步：定义数据来源与计算逻辑。

# 2. 当前上下文
- 指标名称: '{indicator_name}'
- 业务域: '{business_domain}'
- 当前步骤: 第 4 步 - 数据来源
- 已完成步骤信息:
  - 指标定义/计算公式 (第2步): '{step2_output}'
  - 分析维度 (第3步): '{step3_output}'
- 用户的最新输入: '{user_input}'

# 3. 你的核心任务：智能推荐数据源并生成示例SQL
你必须严格遵循以下决策流程：

**第一步：分析上下文 (MANDATORY)**
- **行动**: 回顾并深入理解第2步的「计算公式」和第3步的「分析维度」，提取计算指标所需的核心**度量**（如：产量、工时、不良品数）和**维度**（如：产品线、车间、时间、班次）。

**第二步：获取表和字段推荐 (MANDATORY)**
- **行动**: 基于上一步提取的度量和维度，调用 `get_table_and_field_info` 工具来获取最相关的数据表和字段建议。
- **参数**:
  - `indicator_name`: '{indicator_name}'
  - `formula`: '{step2_output}'  // 将第2步的输出作为 formula 参数
  - `dimensions`: '{step3_output}' // 将第3步的输出作为 dimensions 参数
- **目的**: 获取用于构建数据来源和SQL的候选数据结构，确保推荐的专业性和准确性。

**第三步：决策与响应**
- **如果工具调用成功 (`success: True`)**:
  - **任务**: 利用工具返回的内容，生成包含两部分内容的专业建议：**数据来源**和**SQL示例**。
  - **行动**:
    1.  **生成「数据来源」**: 清晰地描述数据来源于哪些表，并说明关键字段的业务含义。
    2.  **生成「SQL示例」**: 编写一段高质量、可读性强的SQL查询代码，该SQL必须能体现计算逻辑，并包含分析维度。
    3.  **呈现给用户**: 将上述两部分内容整合后，清晰地呈现给用户，并询问他们是否接受。
  - **示例**: "根据您定义的指标和分析视角，我推荐以下数据源和计算逻辑：...（此处省略示例）...请问这个版本是否符合您的要求？"

- **如果工具调用失败 (`success: False`)**:
  - **任务**: 基于你对制造业数据模型的通用理解，独立生成数据来源和SQL示例。
  - **行动**:
    1.  **创建假设**: 明确告知用户，由于无法自动推荐，你将基于行业通用数据模型进行设计。
    2.  **生成内容**: 按照上述成功情况下的格式，生成「数据来源」和「SQL示例」。
    3.  **呈现给用户**: 向用户清晰地说明这是系统生成的通用建议，并请求他们确认或提供其真实环境的表名和字段名。
"""

# 【优化】将 AgentState 移到顶部，因为它被多个函数引用
class AgentState(TypedDict):
    """
    【优化】为每个字段添加了更清晰的注释，特别是 messages 和 history_messages 的区别。
    """
    # UI展示的对话消息列表 (HumanMessage, AIMessage)
    messages: List[BaseMessage]

    # 提供给LLM的完整上下文，包括系统提示、工具消息等
    history_messages: List[BaseMessage]

    # 当前所处的步骤编号 (1-8)，0代表尚未开始
    current_step: int
    # 指标名称
    indicator_name: str
    # 业务域
    business_domain: Optional[str]
    # 业务域的允许范围
    allowed_business_domains: List[str]
    # 存储已完成步骤的数据
    card_data: Dict[int, Dict]
    # 最新一次的用户输入字符串
    user_input: str
    # 错误信息
    error_message: Optional[str]
    # 会话相关ID
    user_id: Optional[str]
    thread_id: Optional[str]
    session_id: str


# 加载环境变量
load_dotenv()


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
logger.info("所有库已成功导入。")


# 为 get_report_prompt 工具定义输入模型
class GetReportPromptArgs(BaseModel):
    indicator_name: str = Field(description="需要查询的指标名称")
    indicator_type: str = Field(description="指标的类型，例如 '名词定义'")
    business_domain: str = Field(description="指标所属的业务领域")

# ==============================================================================
# 2. 工具定义 (保持不变)
# ==============================================================================


@tool(args_schema=GetReportPromptArgs)
def get_report_prompt(indicator_name: str, indicator_type: str, business_domain: str) -> dict:
    """根据指标名称、类型和业务域，优先查询私有库提示词，查不到再查云端提示词。"""
    logger.info(f"\n--- 正在调用工具 [get_report_prompt] ---")
    logger.info(
        f"参数: indicator_name='{indicator_name}', indicator_type='{indicator_type}', business_domain='{business_domain}'")

    # 1. 先查私有库
    private_result = asyncio.run(query_private_prompts(indicator_name, indicator_type, business_domain))
    if private_result.get("success"):

        # 需要判断是否真的有内容
        if not private_result.get("query_result"):
            logger.info("私有库查询成功，但结果为空，继续查云端。")
        else:
            logger.info("私有库查询成功，返回结果。")
            return {
                "success": True,
                "source": "私有库",
                "data": private_result.get("query_result", {})
            }


    # 2. 再查云端
    cloud_result = asyncio.run(query_cloud_prompts(indicator_name, indicator_type, business_domain))
    if cloud_result.get("success"):

        # 同样需要判断是否真的有内容
        if not cloud_result.get("query_result"):
            logger.info("云端查询成功，但结果为空，最终返回未找到。")
        else:
            logger.info("云端查询成功，返回结果。")
            return {
                "success": True,
                "source": "御策云端",
                "data": cloud_result.get("query_result", {})
            }


    # 3. 都查不到
    return {
        "success": False,
        "source": "N/A",
        "data": {},
        "error": "私有库和云端均未查到相关提示词"
    }


# 为 save_or_update_prompt 工具定义输入模型
class SaveOrUpdatePromptArgs(BaseModel):
    title: Optional[str] = Field(None, description="指标/提示词标题")
    type: Optional[str] = Field(None, description="指标类型")
    fields: Optional[str] = Field(None, description="业务域")
    instruction: Optional[str] = Field(None, description="用法说明")
    inputs: Optional[str] = Field(None, description="输入参数示例")
    sql_example: Optional[str] = Field(None, description="SQL示例")

@tool
def save_or_update_prompt(step: int, prompt_object: SaveOrUpdatePromptArgs) -> dict:
    """将某一个步骤已确认的提示词内容保存或更新到用户的私有库。"""
    logger.info(f"\n--- 正在调用工具 [save_or_update_prompt] ---")
    # ...

    # 打印prompt_object对象
    logger.info(f"参数: step={step}, prompt_object={prompt_object}")
    # 参数值类似：step=1, prompt_object=title='生产计划达成率（周）' type='名词定义' fields='生产' instruction='生
    # 计划达成率是衡量企业实际生产完成情况与计划目标匹配程度的指标，用于评估生产系统的执行效率和可控性。' inputs='' sql_example=''
    title = prompt_object.title or ""
    type = prompt_object.type or ""
    fields = prompt_object.fields or ""
    instruction = prompt_object.instruction or ""
    inputs = prompt_object.inputs or ""
    sql_example = prompt_object.sql_example or ""

    # 执行保存或更新逻辑
    result = asyncio.run(save_or_update_prompts(title, type, fields, instruction, inputs, sql_example))

    if not result.get("success"):
        return {
            "success": False,
            "message": f"第 {step} 步的数据保存失败: {result.get('error', '未知错误')}",
            "control": {
                "next_step": step,  # 保持在当前步骤
                "final_message": None
            }
        }

    next_step = step + 1
    final_message = "✅ 所有8个步骤已全部完成！感谢您的使用。" if next_step > 8 else None
    return {
        "success": True,
        "message": f"第 {step} 步的数据已成功保存到私有库！",
        "control": {
            "next_step": next_step,
            "final_message": final_message
        }
    }


@tool
def get_table_and_field_info(indicator_name: str, formula: str = "", dimensions: str = "") -> dict:
    """
    根据指标名称、计算公式和分析维度，查询并推荐相关的数据库表和字段。
    """
    import asyncio
    from src.tools.tableInfoQuery import AgentState, get_columns_node

    logger.info(f"\n--- 正在调用工具 [get_table_and_field_info] ---")
    logger.info(f"参数: indicator_name='{indicator_name}', formula='{formula}', dimensions='{dimensions}'")

    # 构造推荐参数
    prompt_info = f"指标名称: {indicator_name}\n计算公式: {formula}\n分析维度: {dimensions}"

    # 构造 AgentState 类型参数
    state: AgentState = {
        "prompt_info": prompt_info,
        "all_table_comments": asyncio.run(get_all_table_comments_tool.ainvoke({}))
    }

    # 异步调用推荐函数
    result = asyncio.run(recommend_tables_node(state))
    logger.info(f"推荐结果: {result}")
    analysis = result.get("recommendation_analysis")
    if result.get("success") and analysis and analysis.recommended_tables:
        # 获取字段信息
        table_columns = asyncio.run(get_table_columns_tool.ainvoke({"tables": analysis.recommended_tables}))
        return {
            "success": True,
            "recommended_tables": analysis.recommended_tables,
            "table_columns": table_columns,
            "recommendation_reason": analysis.reason
        }
    else:
        return {
            "success": False,
            "recommended_tables": [],
            "table_columns": "",
            "recommendation_reason": result.get("error_message", "未能推荐相关表和字段")
        }

@tool
def generate_sql(natural_language_prompt: str, indicator_name: str, step: int) -> dict:
    """根据用户的自然语言需求、指标上下文，生成SQL查询。用于第5步和第6步。"""
    logger.info(f"\n--- 正在调用工具 [generate_sql] ---")
    logger.info(f"参数: natural_language_prompt='{natural_language_prompt}', step={step}")
    if step == 5:
        sql = "-- 趋势分析SQL示例\nSELECT year_week, SUM(actual_qty) / SUM(plan_qty) FROM dm_prod_plan_actual GROUP BY year_week;"
    else:
        sql = "-- 明细查询SQL示例\nSELECT * FROM t_prod_order_item WHERE order_id = '...';"
    return {"success": True, "generated_sql": sql}


tools = [get_report_prompt, save_or_update_prompt, get_table_and_field_info, generate_sql]
logger.info("工具定义完成。")

# ==============================================================================
# 3. 初始化模型 (保持不变)
# ==============================================================================
try:
    merged_conf = {'api_key': AppConfig.LLM_API_KEY,
                   'base_url': AppConfig.LLM_BASE_URL,
                   'extra_body': {'enable_thinking': False}, 'max_retries': 3, 'model': AppConfig.LLM_MODEL_NAME}
    llm = ChatDashscope(**merged_conf)
    llm_with_tools = llm.bind_tools(tools)
    logger.info("通义千问模型 (qwen-max) 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败，请检查API KEY。错误: {e}")
    exit()


# ==============================================================================
# 4. Agent核心逻辑 (节点函数)
# ==============================================================================

def get_system_prompt(state: AgentState) -> str:
    """根据当前状态动态生成系统提示。"""
    step = state.get("current_step", 0)
    if step == 0:
        return SYSTEM_PROMPT_TEMPLATE_START

    # 【MODIFIED】当进入第四步时，使用专用的、更强大的提示词模板
    if step == 4:
        logger.info("当前是第4步，生成专用的数据来源提示词。")
        card_data = state.get("card_data", {})

        # 从 card_data 中获取第2步和第3步的输出，如果不存在则提供默认值
        # 注意：这里的键（如'instruction'）取决于您在 card_data 中存储的格式
        step2_output = card_data.get(2, {}).get('instruction', '未提供')
        step3_output = card_data.get(3, {}).get('instruction', '未提供')

        return SYSTEM_PROMPT_TEMPLATE_STEP_4.format(
            indicator_name=state.get("indicator_name", "未指定"),
            business_domain=state.get("business_domain", "未指定"),
            step2_output=step2_output,
            step3_output=step3_output,
            user_input=state.get('user_input', '')
        )

    # 对于其他步骤，使用通用模板
    step_name = AppConfig.STEP_MAP.get(step, "未知")
    logger.info(f"生成系统提示: 当前步骤 {step} - {step_name}")
    return SYSTEM_PROMPT_TEMPLATE_STEP.format(
        indicator_name=state.get("indicator_name", "未指定"),
        business_domain=state.get("business_domain", "未指定"),
        step=step,
        step_name=step_name,
        user_input=state.get('user_input', '')
    )


async def get_business_domain_by_llm(indicator_name: str) -> str:
    """根据指标名称判断业务域。"""
    prompt = f"请根据指标名称“{indicator_name}”，在“生产、采购、销售、供应链”中选择最合适的业务域，只输出业务域本身。"
    # 使用 asyncio.to_thread 包装同步的 invoke 调用
    result = await asyncio.to_thread(
        llm.invoke,
        prompt
    )
    allowed = ["生产", "采购", "销售", "供应链"]
    for domain in allowed:
        if domain in result.content:
            return domain
    return "生产"


def ensure_state_fields(state: dict) -> dict:
    defaults = {
        "current_step": 0,
        "indicator_name": "",
        "business_domain": None,
        "allowed_business_domains": AppConfig.ALLOWED_BUSINESS_DOMAINS,
        "card_data": {},
        "history_messages": [],
        "user_input": "",
        "error_message": None,
        "next_action": "",
        "tool_call": None,
        "user_id": None,
        "thread_id": None,
        "session_id": None,
        "messages": [],
    }
    for k, v in defaults.items():
        if k not in state:
            state[k] = v

    if state["session_id"] is None:
        import uuid
        state["session_id"] = str(uuid.uuid4())

    return state


def _normalize_messages(messages: List[Dict | BaseMessage]) -> List[BaseMessage]:
    """将消息列表统一转换为BaseMessage对象列表。"""
    processed = []
    if not messages:
        return []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            processed.append(msg)
            continue
        if isinstance(msg, dict):
            msg_type = msg.get("type")
            content = msg.get("content", "")
            # 处理 {'content': [{'type': 'text', 'text': '...'}]} 的情况
            if isinstance(content, list) and content and "text" in content[0]:
                text_content = content[0]["text"]
            else:
                text_content = str(content)

            if msg_type == "human":
                processed.append(HumanMessage(content=text_content))
            elif msg_type in ["ai", "assistant"]:
                processed.append(AIMessage(content=text_content))
    return processed


async def _handle_flow_start(state: AgentState) -> AgentState:
    """处理“新增”或“修改”指令，启动8步流程。"""
    user_input = state["user_input"]
    logger.info(f"检测到流程启动指令: '{user_input}'")

    # 提取指标名称
    indicator_name = user_input.replace("新增", "").replace("修改", "").strip()
    if not indicator_name:
        # 如果用户只说了“新增”，但没说指标，可以进行反问
        ai_message = AIMessage(content="好的，您想新增哪个指标呢？请输入指标的名称。")
        state["messages"].append(ai_message)
        state["history_messages"].append(ai_message)
        return state

    business_domain = await get_business_domain_by_llm(indicator_name)

    state["current_step"] = 1
    state["indicator_name"] = indicator_name
    state["business_domain"] = business_domain

    ai_content = f"好的，我们开始创建「{indicator_name}」的分析卡片。现在是第1步：名词定义。"
    ai_message = AIMessage(content=ai_content)

    # 为了让流程继续，我们调用LLM获取第一步的建议
    # 注意：这里我们构造一个新的临时状态来生成提示词
    first_step_state = state.copy()
    first_step_state["user_input"] = ""  # 清空输入，避免干扰第一步的引导
    first_step_prompt = get_system_prompt(first_step_state)

    first_step_response = await asyncio.to_thread(
        llm_with_tools.invoke,
        first_step_prompt
    )

    state["messages"].extend([ai_message, first_step_response])
    state["history_messages"].extend([ai_message, first_step_response])

    logger.info(f"操作类型: 新增/修改指标, 指标名称: '{indicator_name}', 业务域: '{business_domain}'")
    return state

from src.tools.tableInfoQuery import recommend_tables_node, get_all_table_comments_tool, get_table_columns_tool


# 专门的函数来调用推荐表和字段的工具
async def get_table_recommendation(prompt_info: str):

    from src.tools.tableInfoQuery import AgentState
    # 构造 AgentState 类型参数
    state: AgentState = {
        "prompt_info": prompt_info,
        "all_table_comments": asyncio.run(get_all_table_comments_tool.ainvoke({}))
    }

    # 调用推荐函数，参数信息为：
    logger.info(f"调用推荐函数，参数信息为：{state}")

    result = await recommend_tables_node(state)

    logger.info(f"推荐结果：{result}")
    return result.get("recommendation_analysis")

async def agent_node(state: AgentState, config: RunnableConfig) -> Coroutine[Any, Any, AgentState] | dict:
    """
    【核心优化】这是Agent的主要思考节点。
    1. 确保每次都从最新的`state`开始。
    2. 将用户的输入（HumanMessage）正确添加到`messages`和`history_messages`。
    3. 调用LLM后，将AI的回复（AIMessage）也添加到这两个列表中。
    4. 返回完整的、更新后的状态。
    """
    logger.info("\n--- 进入 [agent_node] ---")

    # 确保 state 包含所有默认字段
    state = ensure_state_fields(state)

    # 确保 history_messages 存在
    if "history_messages" not in state or state["history_messages"] is None:
        state["history_messages"] = []

    state["messages"] = _normalize_messages(state.get("messages", []))

    # ========================== [END] 新增：数据清洗步骤 ============================

    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):

        logger.info("检测到新的用户输入，更新 user_input 和 history_messages。")

        last_human_message = state["messages"][-1]
        # 同步到 history_messages (如果它还没被添加的话)
        if not state["history_messages"] or state["history_messages"][-1] != last_human_message:
            state["history_messages"].append(last_human_message)

        # 更新当前的用户输入
        if isinstance(last_human_message.content, str):
            state["user_input"] = last_human_message.content
        else:
            state["user_input"] = str(last_human_message.content)
    else:
        logger.info("没有新的用户输入，保持现有的 user_input 不变。")
        # 如果最后一条消息不是HumanMessage，意味着我们是从工具调用返回的，没有新的用户输入。
        state["user_input"] = ""
    # ========================== [END] 修改后的逻辑 ============================

    # 获取用户输入
    user_input = state.get("user_input", "")
    if isinstance(user_input, list):
        # 如果是列表，拼接为字符串
        user_input = " ".join(map(str, user_input))
    elif not isinstance(user_input, str):
        # 如果不是字符串，转换为字符串
        user_input = str(user_input)

    user_input_lower = user_input.lower()
    state["user_input"] = user_input  # 确保更新后的值存储回 state
    step = state.get("current_step", 0)

    logger.info(f"当前步骤: {step}, 用户输入: '{state['user_input']}'")

    # 特殊逻辑：处理流程启动
    if step == 0 and ("新增" in user_input_lower or "修改" in user_input_lower):
        logger.info("检测到新增/修改指令，启动8步流程。")
        return await _handle_flow_start(state)

    # 动态生成系统提示
    system_prompt = get_system_prompt(state)
    # 准备给LLM的消息列表
    llm_messages = [("system", system_prompt)] + state["history_messages"]
    # 通用逻辑：调用LLM进行下一步
    response = await asyncio.to_thread(
        llm_with_tools.invoke,  # 注意：在 to_thread 中我们通常调用同步版本 .invoke
        llm_messages
    )

    # 【优化】这是关键！将AI的回复同时添加到 messages 和 history_messages
    state["messages"].append(response)
    state["history_messages"].append(response)

    logger.info(f"更新后的状态 (agent_node): messages count = {len(state['messages'])}")
    return state


# 在工具定义后创建映射
tool_map = {t.name: t for t in tools}


async def tool_node(state: AgentState) -> AgentState:
    """
    【核心优化】工具执行节点。
    1. 执行工具调用。
    2. 将工具结果（ToolMessage）只添加到`history_messages`，因为UI不需要展示这个。
    3. `messages`列表保持不变。
    4. 返回完整的、更新后的状态。
    """
    logger.info("\n--- 进入 [tool_node] ---")
    last_message = state["history_messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        state["error_message"] = "Agent节点未请求调用工具，但被路由到了tool_node。"
        return state

    all_tool_messages = []
    current_step = state.get("current_step", 0) # 获取当前步骤

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        tool_params = tool_call["args"]
        # 使用 .get() 进行更安全的查找
        target_tool = tool_map.get(tool_name)

        if not target_tool:
            error_message = f"找不到名为 {tool_name} 的工具"
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))
            continue

        # 【新增】在调用 save_or_update_prompt 前，将数据存入 state
        if tool_name == "save_or_update_prompt":
            logger.info(f"检测到保存操作，正在将第 {current_step} 步的数据更新到 state['card_data']。")

            # 从工具参数中提取需要保存的内容
            # prompt_object 是 SaveOrUpdatePromptArgs 模型实例的字典表示
            prompt_data_to_save = tool_params.get('prompt_object', {})

            # 确保 card_data 存在
            if 'card_data' not in state:
                state['card_data'] = {}

            # 将数据存入 card_data
            state['card_data'][current_step] = prompt_data_to_save
            logger.info(f"更新后的 card_data: {state['card_data']}")

        try:
            result = await asyncio.to_thread(
                target_tool.invoke,  # 同样使用同步版本的 .invoke
                tool_params
            )
            all_tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

            # 数据驱动的流程控制
            if isinstance(result, dict) and "control" in result:
                control_info = result["control"]
                if "next_step" in control_info:
                    state["current_step"] = control_info["next_step"]
                    logger.info(f"流程已由工具推进到第 {state['current_step']} 步。")

        except Exception as e:
            error_message = f"执行工具 {tool_name} 时出错: {e}"
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))

    # 确保所有工具响应消息都添加到 history_messages
    state["history_messages"].extend(all_tool_messages)

    # 验证是否所有 tool_call_id 都有响应
    tool_call_ids = {call["id"] for call in last_message.tool_calls}
    responded_ids = {msg.tool_call_id for msg in all_tool_messages if hasattr(msg, "tool_call_id")}
    missing_ids = tool_call_ids - responded_ids

    if missing_ids:
        state["error_message"] = f"以下 tool_call_id 缺少响应消息: {missing_ids}"
        logger.info(state["error_message"])

    logger.info(f"更新后的状态 (tool_node): messages count = {len(state['messages'])}")
    return state


def should_continue(state: AgentState) -> str:
    """路由函数，决定下一步是调用工具还是结束。"""
    # 确保 history_messages 存在
    if "history_messages" not in state or not state["history_messages"]:
        state["history_messages"] = []

    last_message = state["history_messages"][-1] if state["history_messages"] else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("\n--- [路由决策] -> call_tool ---")
        return "call_tool"
    else:
        # 如果AI的回复中包含“所有8个步骤已全部完成”，也可以结束
        if isinstance(last_message, AIMessage) and "所有8个步骤已全部完成" in last_message.content:
            logger.info("\n--- [路由决策] -> end (流程完成) ---")
            return END
        logger.info("\n--- [路由决策] -> end (等待用户输入) ---")
        return END


logger.info("Graph节点定义完成。")

# ==============================================================================
# 5. 构建并编译图 (Graph)
# ==============================================================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        # 【优化】直接使用END常量，更标准
        END: END
    }
)
workflow.add_edge("call_tool", "agent")

# 【优化】默认使用内存存储，如果需要数据库持久化，再取消注释相关代码
# checkpointer = MemorySaver()
# manufacturing_assistant = workflow.compile(checkpointer=checkpointer)
manufacturing_assistant = workflow.compile()
logger.info("✅ Agent图编译完成。")


# ==============================================================================
# 6. 主函数 (为本地测试保留，实际部署时会被FastAPI等框架替代)
# ==============================================================================

def create_new_state(session_id: Optional[str] = None) -> AgentState:
    """创建一个新的、完整的初始AgentState。"""
    session_id = session_id or str(uuid.uuid4())
    return AgentState(
        messages=[],
        history_messages=[],
        current_step=0,
        indicator_name="",
        business_domain=None,
        allowed_business_domains=AppConfig.ALLOWED_BUSINESS_DOMAINS,
        card_data={},
        user_input="",
        error_message=None,
        user_id="local_user",  # or dynamically set
        thread_id=session_id,
        session_id=session_id
    )


async def main():
    logger.info("\n==================================================")
    logger.info("你好！我是「制造业指标分析卡片助手」")
    logger.info("您可以输入'新增 [指标名称]'来开始创建分析卡片。")
    logger.info("输入 'exit' 或 Ctrl+C 来退出程序。")
    logger.info("==================================================")

    session_id = str(uuid.uuid4())
    logger.info(f"本次会话ID: {session_id}")

    config = {"configurable": {"thread_id": session_id}}

    initial_state = create_new_state(session_id)

    # 模拟一次UI交互
    while True:
        try:
            user_input = input(f"\n你 [{session_id[:8]}]> ")
            if user_input.lower() in ["exit", "退出"]:
                break

            # 【优化】这是与UI交互的正确模式：
            # 1. 将用户的新消息添加到当前状态的 `messages` 列表中。
            # 2. 将这个包含新消息的列表作为输入，调用stream方法。
            # LangGraph 会自动将这个输入合并到现有的状态中。
            inputs = {"messages": [HumanMessage(content=user_input)]}

            # 流式处理响应
            final_state = None
            async for event in manufacturing_assistant.stream(inputs, config=config, stream_mode="values"):
                final_state = event

            if final_state and final_state["messages"]:
                last_message = final_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    logger.info(f"助手> {last_message.content}")

        except KeyboardInterrupt:
            logger.info("\n再见！")
            break
        except Exception as e:
            logger.error(f"\n程序发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    # 当你将此文件作为服务部署时，main() 函数通常不会被调用。
    # API框架（如FastAPI）会直接导入和使用 `manufacturing_assistant` 这个编译好的图。
    import asyncio

    # 使用 asyncio.run() 来运行异步主函数
    asyncio.run(main())
    # main()
    # pass
