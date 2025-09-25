# -*- coding: utf-8 -*-
# manufacturing_agent.py

import os
import sys
import uuid
import operator
from typing_extensions import TypedDict, List, Dict, Optional, Annotated

# ==============================================================================
# 0. 环境和依赖导入
# ==============================================================================
# 使用 python-dotenv 库来加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv()

# LangChain 和 LangGraph 的核心组件
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
# from langchain_community.chat_models.tongyi import QianwenChat
from langgraph.graph import StateGraph, END
#from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
# from src.llms.llm import get_llm_by_type
from src.tools.dashscope import ChatDashscope
from langchain_core.runnables import RunnableConfig


# 解决控制台中文乱码问题
def setup_encoding():
    try:
        if os.name == 'nt': os.system('chcp 65001 > nul')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        print(f"编码设置失败: {e}")


setup_encoding()

print("所有库已成功导入。")


# ==============================================================================
# 1. Agent状态定义 (State)
# ==============================================================================
class AgentState(TypedDict):
    """
    定义了Agent在整个对话流程中的状态。
    这个状态对象会在图的每个节点之间传递和更新。
    """
    # 当前所处的步骤编号 (1-8)，0代表尚未开始
    current_step: int

    # 指标名称，例如 "生产计划达成率（周）"
    indicator_name: str

    # 业务域，在第一步确认后，后续步骤将沿用
    business_domain: Optional[str]

    # 【新增】定义业务域的允许范围
    allowed_business_domains: List[str]

    # 存储已完成步骤的数据
    # 格式: {1: {"title": "...", "type": "名词定义", ...}, 2: {...}}
    card_data: Dict[int, Dict]

    # 完整的对话历史记录
    history_messages: List[BaseMessage]

    # 最新一次的用户输入
    user_input: str

    # 如果在执行工具或节点时发生错误，错误信息会存放在这里
    error_message: Optional[str]

    # Supervisor节点决策出的下一步动作 (例如 'call_tool', 'process_step', 'end')
    next_action: str

    # 如果需要调用工具，这里存放工具调用的相关信息
    tool_call: Optional[Dict]

    user_id: Optional[str]      # 新增
    thread_id: Optional[str]    # 新增
    session_id: str = None              # 会话ID，用于标识一次完整的对话
    messages: List[BaseMessage] = None  # 聊天消息历史列表


print("AgentState 定义完成。")


# ==============================================================================
# 2. 工具定义 (Tools)
# ==============================================================================
# 【重要】请在此区域填充您真实的业务逻辑。
# 以下所有函数均为占位符，用于演示Agent的运行流程。

@tool
def get_report_prompt(indicator_name: str, indicator_type: str, business_domain: str) -> dict:
    """
    根据指标名称、类型和业务域，查询私有库或御策云端的标准提示词。
    """
    print(f"\n--- 正在调用工具 [get_report_prompt] ---")
    print(
        f"参数: indicator_name='{indicator_name}', indicator_type='{indicator_type}', business_domain='{business_domain}'")

    # 【TODO】: 在这里替换为查询您数据库或API的真实逻辑
    if indicator_type == "名词定义":
        return {
            "success": True,
            "source": "御策云端",
            "data": {"用途说明": "生产计划达成率是衡量企业实际生产完成情况与计划目标匹配程度的指标..."}
        }
    return {"success": False, "source": "N/A", "data": {}}


@tool
def save_or_update_prompt(step: int, prompt_object: dict) -> dict:
    """
    将某一个步骤已确认的提示词内容保存或更新到用户的私有库。
    """
    print(f"\n--- 正在调用工具 [save_or_update_prompt] ---")
    print(f"参数: step={step}, prompt_object={prompt_object}")

    # 【TODO】: 在这里替换为写入您数据库的真实逻辑
    return {"success": True, "message": f"第 {step} 步的数据已成功保存到私有库！"}


@tool
def get_table_and_field_info(indicator_name: str, formula: str = "", dimensions: str = "") -> dict:
    """
    根据指标名称、计算公式和分析维度，查询并推荐相关的数据库表和字段。
    """
    print(f"\n--- 正在调用工具 [get_table_and_field_info] ---")
    print(f"参数: indicator_name='{indicator_name}'")

    # 【TODO】: 在这里替换为查询您元数据系统的真实逻辑
    return {
        "success": True,
        "recommended_tables": [
            {"table_name": "t_prod_order_item", "reason": "包含计划产量(PlanQty)和实际产量(ActualQty)"}
        ],
        "recommendation_reason": "建议使用 t_prod_order_item 表进行分析。"
    }


@tool
def generate_sql(natural_language_prompt: str, indicator_name: str, step: int) -> dict:
    """
    根据用户的自然语言需求、指标上下文，生成SQL查询。用于第5步和第6步。
    """
    print(f"\n--- 正在调用工具 [generate_sql] ---")
    print(f"参数: natural_language_prompt='{natural_language_prompt}', step={step}")

    # 【TODO】: 在这里可以调用一个更专业的Text-to-SQL模型或服务
    if step == 5:
        sql = "-- 趋势分析SQL示例\nSELECT year_week, SUM(actual_qty) / SUM(plan_qty) FROM dm_prod_plan_actual GROUP BY year_week;"
    else:
        sql = "-- 明细查询SQL示例\nSELECT * FROM t_prod_order_item WHERE order_id = '...';"
    return {"success": True, "generated_sql": sql}


tools = [get_report_prompt, save_or_update_prompt, get_table_and_field_info, generate_sql]
print("工具定义完成。")

# ==============================================================================
# 3. 初始化模型
# ==============================================================================
try:
    merged_conf = {'api_key': 'sk-ca949c46e4904479927923a41562d4d3',
                   'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                   'extra_body': {'enable_thinking': False}, 'max_retries': 3, 'model': 'qwen-max-latest'}
    llm = ChatDashscope(**merged_conf)
    # llm = get_llm_by_type("basic")
    llm_with_tools = llm.bind_tools(tools)
    print("通义千问模型 (qwen-max) 初始化成功。")
except Exception as e:
    print(f"模型初始化失败，请检查DASHSCOPE_API_KEY环境变量是否正确设置。错误: {e}")
    exit()


def get_system_prompt(state: AgentState) -> str:
    """【新增】根据当前状态动态生成系统提示，让LLM更清楚自己的任务。"""
    step = state.get("current_step", 0)
    indicator_name = state.get("indicator_name", "未指定")

    if step == 0:
        return "你是「制造业指标分析卡片助手」。你的任务是友好地响应用户，并在用户输入'新增'或'修改'指令时，开始8步引导流程。"

    step_map = {
        1: "名词定义", 2: "计算公式", 3: "分析维度与指标", 4: "数据来源",
        5: "指标计算-结果查询SQL", 6: "明细计算-实时计算SQL",
        7: "指标分析方法", 8: "明细分析方法"
    }
    step_name = step_map.get(step, "未知")

    return f"""你是「制造业指标分析卡片助手」。你正在严格按照8步流程引导用户。
当前状态：
- 指标名称: '{indicator_name}'
- 正在进行的步骤: 第 {step} 步 - {step_name}

你的任务是：
1. 分析用户的最新输入 (`{state['user_input']}`)。
2. 如果用户确认上一步的内容（例如说“是”或“好”），调用 `save_or_update_prompt` 工具保存数据，然后推进到下一步。
3. 如果用户是否定或提供了自定义内容，也调用 `save_or_update_prompt` 保存，然后推进。
4. 在新的一步，生成引导性内容，并询问用户是否采纳。你可以调用 `get_report_prompt` 等工具来获取信息。
5. 严格遵守你的角色，不要偏离8步流程。"""


# 1. 第一步时，让LLM判断业务域
def get_business_domain_by_llm(indicator_name: str) -> str:
    prompt = (
        f"请根据指标名称“{indicator_name}”，在“生产、采购、销售、供应链”中选择最合适的业务域，只输出业务域本身。"
    )
    result = llm.invoke(prompt)
    # 简单处理，确保只返回四选一
    allowed = ["生产", "采购", "销售", "供应链"]
    for domain in allowed:
        if domain in result.content:
            return domain
    return "生产"  # 默认兜底


def ensure_state_fields(state: dict) -> dict:
    defaults = {
        "current_step": 0,
        "indicator_name": "",
        "business_domain": None,
        "allowed_business_domains": ["生产", "采购", "销售", "供应链"],
        "card_data": {},
        "history_messages": [],
        "user_input": "你好啊",
        "error_message": None,
        "next_action": "",
        "tool_call": None,
        "user_id": None,        # 新增
        "thread_id": None,       # 新增
        "session_id": None,  # 默认会话ID
        "messages": [],
    }
    for k, v in defaults.items():
        if k not in state:
            state[k] = v


    # 如果没有会话ID，生成一个默认的UUID
    if state["session_id"] is None:
        import uuid
        state["session_id"] = str(uuid.uuid4())

    return state


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """【修改】核心Agent节点，合并了原supervisor和process_step的职责。"""
    print("\n--- 进入 [agent_node] ---")

    state = ensure_state_fields(state)  # 确保所有字段存在

    # 从config中获取user_id和thread_id
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("configurable", {}).get("user_id")
    if thread_id:
        state["thread_id"] = thread_id
    if user_id:
        state["user_id"] = user_id

    # 打印线程和用户信息，还有session_id
    print(f"当前线程ID: {state.get('thread_id')}, 用户ID: {state.get('user_id')}, 会话ID: {state.get('session_id')}")

    system_prompt = get_system_prompt(state)
    messages = [("system", system_prompt)] + state["history_messages"]

    user_input_lower = state.get("user_input", "").lower()
    step = state.get("current_step", 0)

    # 2. agent_node 启动流程时，调用LLM判断业务域
    if (("新增" in user_input_lower or "修改" in user_input_lower)) and step == 0:
        indicator_name = state["user_input"].replace("新增", "").replace("修改", "").strip()
        new_step = 1
        business_domain = get_business_domain_by_llm(indicator_name)
        ai_message = AIMessage(content=f"好的，我们开始创建「{indicator_name}」的分析卡片。现在是第1步：名词定义。")
        first_step_prompt = get_system_prompt({
            "current_step": new_step,
            "indicator_name": indicator_name,
            "user_input": "",
            "business_domain": business_domain,
            "allowed_business_domains": ["生产", "采购", "销售", "供应链"]
        })
        first_step_response = llm_with_tools.invoke(first_step_prompt)

        # 打印操作类型和操作信息
        print(f"操作类型: 新增指标，指标名称: '{indicator_name}'，业务域: '{business_domain}'")
        # 打印state所有信息
        print(f"更新后的状态: {state}")

        return {
            "current_step": new_step,
            "indicator_name": indicator_name,
            "business_domain": business_domain,
            "allowed_business_domains": ["生产", "采购", "销售", "供应链"],
            "card_data": {},
            "session_id": state["session_id"],
            "user_id": state["user_id"],   # 添加用户ID
            "thread_id": state["thread_id"],
            "history_messages": state["history_messages"] + [ai_message, first_step_response],
            "messages": state["messages"] + [ai_message, first_step_response],
        }

    response = llm_with_tools.invoke(messages)

    new_state = state.copy()
    new_state["history_messages"] = state["history_messages"] + [response]
    # 打印state所有信息
    print(f"更新后的状态: {new_state}")
    return new_state

def tool_node(state: AgentState) -> dict:
    """执行工具的节点，确保返回的消息格式正确。"""
    print("\n--- 进入 [tool_node] ---")
    last_message = state["history_messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {
            "error_message": "Agent节点未请求调用工具，但被路由到了tool_node。",
            "messages": state["history_messages"],
            "session_id": state["session_id"],
            "user_id": state["user_id"],
            "thread_id": state["thread_id"],
        }

    tool_calls = last_message.tool_calls
    all_tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_params = tool_call["args"]
        target_tool = next((t for t in tools if t.name == tool_name), None)

        if not target_tool:
            error_message = f"找不到名为 {tool_name} 的工具"
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
            continue

        try:
            result = target_tool.invoke(tool_params)
            all_tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        except Exception as e:
            error_message = f"执行工具 {tool_name} 时出错: {e}"
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))

    state["history_messages"].extend(all_tool_messages)

    return {
        "messages": state["history_messages"],
        "session_id": state["session_id"],
        "user_id": state["user_id"],
        "thread_id": state["thread_id"],
    }

def tool_node(state: AgentState) -> dict:
    """【修正】执行工具的节点，并正确地返回状态更新。"""
    print("\n--- 进入 [tool_node] ---")
    last_message = state["history_messages"][-1]

    # 添加一个安全检查，防止因意外路由导致错误
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"error_message": "Agent节点未请求调用工具，但被路由到了tool_node。"}

    tool_calls = last_message.tool_calls

    # 【核心修正 1】: 创建一个新的字典来收集所有状态更新
    updates = {}
    all_tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_params = tool_call["args"]
        target_tool = next((t for t in tools if t.name == tool_name), None)

        if not target_tool:
            error_message = f"找不到名为 {tool_name} 的工具"
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
            continue

        try:
            result = target_tool.invoke(tool_params)

            # 【核心修正 2】: 当需要推进步骤时，将更新放入 updates 字典，而不是直接修改 state
            if tool_name == "save_or_update_prompt" and result.get("success"):
                current_step = state.get("current_step", 0)
                next_step = current_step + 1

                # 正确的做法：将 'current_step' 的更新添加到 updates 字典中
                updates["current_step"] = next_step

                # 如果流程结束，可以在工具结果中添加一个最终信息，供agent_node下一步生成回复
                if next_step > 8:
                    result["final_message"] = "✅ 所有8个步骤已全部完成！感谢您的使用。"

            all_tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        except Exception as e:
            error_message = f"执行工具 {tool_name} 时出错: {e}"
            print(f"错误: {error_message}")
            all_tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))

    # 【核心修正 3】: 将更新后的对话历史也放入 updates 字典
    updates["history_messages"] = state["history_messages"] + all_tool_messages

    new_state = state.copy()
    new_state.update(updates)
    return new_state


def should_continue(state: AgentState) -> str:
    """【新增】这是修复问题的关键：一个明确的路由函数，用于决定何时结束循环。"""
    last_message = state["history_messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("\n--- [路由决策] 发现工具调用请求，转到 [call_tool] ---")
        return "call_tool"
    else:
        print("\n--- [路由决策] 未发现工具调用，结束本轮对话 [end] ---")
        return "end"


print("Graph节点定义完成。")

# ==============================================================================
# 5. 构建并编译图 (Graph)
# ==============================================================================
workflow = StateGraph(AgentState)

# 【修改】简化图的节点
workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", tool_node)

# 设置入口节点
workflow.set_entry_point("agent")

# 【修改】使用新的路由函数来控制流程
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        "end": END  # <== 这是停止循环的关键！
    }
)
# 【修改】工具执行完后，必须返回给Agent节点
workflow.add_edge("call_tool", "agent")

# 设置持久化
# db_url = os.getenv("POSTGRES_URL")
# postgres://postgres:kotl2025@192.168.58.75:5432/lobechat?sslmode=disable
# db_url = "postgres://postgres:kotl2025@192.168.58.75:5432/lobechat?sslmode=disable"
#
# if not db_url:
#     print("错误：环境变量 POSTGRES_URL 未设置! 请在 .env 文件中配置。")
#     exit()

try:
    # 自动创建所需的表
    # memory = PostgresSaver.from_url(db_url, setup=True)
    # app = workflow.compile(checkpointer=memory)
    memory = MemorySaver()
    manufacturing_assistant_bak = workflow.compile(checkpointer=memory)
    # manufacturing_assistant_bak = workflow.compile()
    print("✅ Agent图编译完成，已连接到PostgreSQL并设置好持久化。")
except Exception as e:
    print(f"连接PostgreSQL或编译图时出错，请检查连接字符串是否正确。错误: {e}")
    exit()


# manufacturing_assistant = (
#     StateGraph(AgentState)
#     .add_node("agent", agent_node)
#     .add_edge("__start__", "agent")
#     .compile(name="Qwen Chat Agent")
# )

# ==============================================================================
# 6. 主函数 (运行入口)
# ==============================================================================
def main():
    print("\n==================================================")
    print("你好！我是「制造业指标分析卡片助手」。")
    print("您可以输入'新增 [指标名称]'来开始创建分析卡片。")
    print("输入 'exit' 或 Ctrl+C 来退出程序。")
    print("==================================================")

    session_id = str(uuid.uuid4())
    print(f"本次会话ID: {session_id}")

    config = {"configurable": {"thread_id": session_id}}

    # 【修改】移除了这里的 app.update_state 和 initial_state
    print("助手> 你好！我是「制造业指标分析卡片助手」。\n      您可以输入'新增 [指标名称]'来开始创建分析卡片。")

    while True:
        try:
            user_input = input(f"\n你 [{session_id[:8]}]> ")
            if user_input.lower() in ["exit", "退出"]:
                break

            # 【修改】将user_input也放入输入字典中，以便节点使用
            inputs = {"history_messages": [HumanMessage(content=user_input)], "user_input": user_input}

            for event in manufacturing_assistant_bak.stream(inputs, config=config, stream_mode="values"):
                final_state = event

            if final_state and final_state["history_messages"]:
                last_message = final_state["history_messages"][-1]
                if isinstance(last_message, AIMessage):
                    print(f"助手> {last_message.content}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n程序发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
