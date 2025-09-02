"""LangGraph chat agent with context support using Qwen model.

This agent supports contextual conversations and uses the Qwen model from Tongyi.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, TypedDict

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage,SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

#from .config import Config


class Context(TypedDict):
    """Context parameters for the agent.
    
    Set these when creating assistants OR when invoking the graph.
    """
    dashscope_api_key: str


@dataclass
class State:
    """Input state for the agent.
    
    Defines the structure for chat messages and conversation history with context support.
    """
    messages: List[BaseMessage] = None  # 聊天消息历史列表
    session_id: str = None              # 会话ID，用于标识一次完整的对话
    user_id: str = None                 # 用户ID，标识发起对话的用户
    system_prompt: str = (
        "欢迎！我是创建数据同步任务助手。可以协助你完成以下业务：\n"
        "1、导入数据字典和表。导入源数据库的数据字典、表结构到分析库（即analyze_data库）。\n"
        "2、查找业务相关的表。明确同步哪些表到分析库。如：查找**库采购数据相关表。\n"
        "3、创建数据同步任务。"
    )           # 系统提示词，指导AI的行为
    context_data: Dict[str, Any] = None # 上下文数据，存储额外的上下文信息
    metadata: Dict[str, Any] = None     # 元数据，记录会话相关的统计和状态
    thread_id: str = None               # 对话线程ID，用于多线程或多会话场景
    # 定义保存上下文信息
    history_messages: List[BaseMessage] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.history_messages is None:
            print("history_messages is None")
            self.history_messages = []
        if self.context_data is None:
            self.context_data = {}
        if self.metadata is None:
            self.metadata = {}
        
        # 如果没有设置系统提示词，使用默认的
        if self.system_prompt is None:
            self.system_prompt = "你是一个有用的AI助手，基于通义千问模型。请用中文回答用户的问题。"
        
        # 如果没有会话ID，生成一个默认的
        if self.session_id is None:
            import uuid
            self.session_id = str(uuid.uuid4())
        
        # 更新元数据
        self.metadata.update({
            "message_count": len(self.messages),
            "last_updated": None,  # 将在使用时更新
            "context_persisted": True  # 标记上下文已持久化
        })
    
    def add_message(self, message: BaseMessage):
        """添加消息到对话历史并更新元数据"""
        self.messages.append(message)
        self.metadata["message_count"] = len(self.messages)
        from datetime import datetime
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def get_context_summary(self) -> str:
        """获取上下文摘要，用于AI模型理解对话背景"""
        summary_parts = []
        
        if self.user_id:
            summary_parts.append(f"用户ID: {self.user_id}")
        
        if self.thread_id:
            summary_parts.append(f"对话线程: {self.thread_id}")
        
        if self.session_id:
            summary_parts.append(f"会话ID: {self.session_id}")
        
        if self.context_data:
            for key, value in self.context_data.items():
                summary_parts.append(f"{key}: {value}")
        
        return "; ".join(summary_parts) if summary_parts else "无特殊上下文"
    
    def update_context(self, key: str, value: Any):
        """更新上下文数据"""
        self.context_data[key] = value
        from datetime import datetime
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def clear_context(self):
        """清除上下文数据，但保留消息历史"""
        self.context_data.clear()
        self.metadata.clear()
        self.__post_init__()  # 重新初始化元数据


def create_chat_model() -> ChatTongyi:
    """Create and configure the Qwen chat model."""
    #if api_key is None:
    #    api_key = Config.get_api_key()
    
    return ChatTongyi(
        model="qwen-plus", 
        dashscope_api_key="sk-7c61b5435ea94666b3a50d4a0d889bd2"
 
    )


async def chat_with_context(state: State, runtime: Runtime[Context],config: RunnableConfig) -> Dict[str, Any]:
    """Process chat messages with context support.
    
    This function:
    1. Gets the conversation history from state
    2. Uses the Qwen model to generate a response
    3. Maintains context for the conversation
    4. Utilizes context information for better responses
    """
    # 安全地获取API密钥
    #api_key = Config.get_api_key()
    #if runtime.context is not None:
    #    api_key = runtime.context.get('dashscope_api_key', api_key)
    
    # 创建聊天模型
    chat_model = create_chat_model()
    #print(f"config: {config}")
    
    # 从config中获取thread_id并更新到state中
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id:
        state.thread_id = thread_id
    user_id = config.get("configurable", {}).get("user_id")
    if user_id:
        state.user_id = user_id
    
    print(f"thread_id: {state.thread_id}")
    print(f"会话ID: {state.session_id}")
    print(f"用户ID user_id: {state.user_id}")

    # 判断用户ID为空时回复请输入用户ID 用户ID user_id: null
    """ if user_id is None or user_id == "null":
        print("用户ID为空，终止执行。")
        state.messages.append(AIMessage(content="缺少用户ID，请URL中添加user_id参数。"))
        return {"messages": state.messages} """
    
    #print(f"上下文摘要: {state.get_context_summary()}")
    #print(f"消息数量: {len(state.messages)}")
    # 获取用户输入


    # 获取当前会话的消息
    messages = state.messages
    # 打印历史消息
    print(f"历史消息: {messages}")

    
    
    # 处理消息格式转换
    processed_messages = []

    msg = messages[-1]
    #for msg in messages:
    if isinstance(msg, dict):
        # 如果是字典格式，转换为LangChain消息对象
        # [{'id': '3dc8915d-fc02-4a97-a6d2-f9681595e437', 'type': 'human', 'content': [{'type': 'text', 'text': '你好'}], 'additional_kwargs': {'userId': '1234'}}]
        msg_type = msg.get('type', '')
        content = msg.get('content', '')
        user_id = msg.get('additional_kwargs', {}).get('userId', '')
        if user_id:
            state.user_id = user_id
        # 处理复杂的内容格式
        if isinstance(content, list):
            # 如果是列表格式，提取文本内容
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_content += item.get('text', '')
            content = text_content
        elif isinstance(content, str):
            # 如果已经是字符串，直接使用
            pass
        else:
            content = str(content)
        
        if msg_type == 'human':
            print(f"human: {content}")
            processed_messages.append(HumanMessage(content=content,additional_kwargs={'userId': user_id}))
        elif msg_type == 'ai':
            print(f"ai: {content}")
            processed_messages.append(AIMessage(content=content,additional_kwargs={'userId': user_id}))
        else:
            print(f"default: {content}")
            # 默认作为用户消息处理
            processed_messages.append(HumanMessage(content=content,additional_kwargs={'userId': user_id}))
    else:
        # 如果已经是LangChain消息对象，直接使用
        processed_messages.append(msg)

    #state.messages.append(processed_messages)        
    
    print(f"处理后的消息数量: {len(processed_messages)}")
    """ for i, msg in enumerate(processed_messages):
        print(f"消息 {i}: {type(msg).__name__} - {msg.content[:50]}...") """
    
    if not processed_messages:
        # 如果没有消息，返回欢迎信息
        welcome_message = AIMessage(
            content="你好！我是基于通义千问的AI助手。我可以帮助你回答问题，请告诉我你需要什么帮助。"
        )
        # 使用 add_message 方法添加消息并更新元数据
        state.add_message(welcome_message)
        return {"messages": state.messages}
    
    # 构建包含系统提示词和上下文的消息列表
    # 增加系统提示词到上下文消息列表
    
    if state.system_prompt:
        # 检查history_messages中是否已存在SystemMessage，避免重复添加
        has_system_message = any(isinstance(msg, SystemMessage) for msg in state.history_messages)
        if not has_system_message:
            state.history_messages.append(SystemMessage(content=state.system_prompt))
    
    """ # 添加系统提示词
    if state.system_prompt:
        from langchain_core.messages import SystemMessage
        context_messages.append(SystemMessage(content=state.system_prompt))
    
    # 添加上下文信息
    context_summary = state.get_context_summary()
    if context_summary != "无特殊上下文":
        from langchain_core.messages import SystemMessage
        context_messages.append(SystemMessage(content=f"上下文信息: {context_summary}"))
    
    # 添加用户消息
    context_messages.extend(processed_messages)

    print(f"context_messages: {context_messages}") """
    
    # 获取最后一条用户消息
    last_message = processed_messages[-1]
    # 获取当前会话的消息历史
    state.history_messages.append(last_message)
    # 打印历史消息
    print(f"history_messages: {state.history_messages}")
    print(f"last_message: {last_message}")
    
    if isinstance(last_message, HumanMessage):
        # 使用聊天模型生成回复
        try:
            print("开始调用AI模型...")
            print(f"发送给AI的消息数量: {len(processed_messages)}")
            response = await chat_model.ainvoke(state.history_messages)
            print("-----ai msg---------")
            #print(response)
            print(f"response.content: {response.content}")
            
            
            # 使用 add_message 方法添加AI回复并更新元数据
            state.add_message(AIMessage(content=response.content,additional_kwargs={'userId': state.user_id}))
            print(f"----state.messages: {state.messages}")

            state.history_messages.append(AIMessage(content=response.content,additional_kwargs={'userId': state.user_id}))
            print(f"----state.history_messages: {state.history_messages}")
            
            # 返回完整的状态，包括上下文信息
            return {
                "messages": state.messages,
                "history_messages": state.history_messages,
                "session_id": state.session_id,
                "user_id": state.user_id,   # 添加用户ID    
                "thread_id": state.thread_id,
                "context_data": state.context_data,
                "metadata": state.metadata
            }
            
        except Exception as e:
            # 处理错误情况
            print(f"调用AI模型时出错: {e}")
            error_message = AIMessage(
                content=f"抱歉，处理您的消息时出现了错误：{str(e)}。请稍后重试。"
            )
            # 使用 add_message 方法添加错误消息并更新元数据
            state.add_message(error_message)
            return {
                "messages": state.messages,
                "session_id": state.session_id,
                "user_id": state.user_id,
                "thread_id": state.thread_id,
                "context_data": state.context_data,
                "metadata": state.metadata
            }
    
    # 如果不是用户消息，直接返回当前状态
    return {
        "messages": state.messages,
        "session_id": state.session_id,
        "user_id": state.user_id,
        "thread_id": state.thread_id,
        "context_data": state.context_data,
        "metadata": state.metadata
    }

# Create the graph with memory for persistence
#checkpointer = MemorySaver()
# postgresql://username:password@host:port/database?sslmode=disable
#DB_URI = "postgresql://postgres:kotldb%4064@192.168.58.64:5432/langgraph?sslmode=disable"
# 从.env获取 POSTGRES_URI_CUSTOM

#DB_URI = os.getenv("POSTGRES_URI_CUSTOM")

# 设置环境变量
#os.environ["POSTGRES_URI_CUSTOM"] = "postgresql://postgres:kotldb%4064@hb.kotl.cn:5432/langgraph?sslmode=disable"
 # 定义图结构
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("chat_with_context", chat_with_context)
    .add_edge("__start__", "chat_with_context")
    .compile(name="Qwen Chat Agent")
)

""" if os.getenv("LANGGRAPH_DEV") == "true":
    try:
        # 正确的使用方式
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # 首次使用时调用 setup()
            checkpointer.setup()
            print("checkpointer", checkpointer)
            print("DB_URI", DB_URI)
            
            # 定义图结构
            graph = (
                StateGraph(State, context_schema=Context)
                .add_node("chat_with_context", chat_with_context)
                .add_edge("__start__", "chat_with_context")
                .compile(name="Qwen Chat Agent", checkpointer=checkpointer)
            )
            
    except Exception as e:
        print(f"数据库连接失败: {e}")
        print("回退到内存存储模式")
        # 回退到内存存储
        graph = (
            StateGraph(State, context_schema=Context)
            .add_node("chat_with_context", chat_with_context)
            .add_edge("__start__", "chat_with_context")
            .compile(name="Qwen Chat Agent")
        )
       
else:
   graph = (
            StateGraph(State, context_schema=Context)
            .add_node("chat_with_context", chat_with_context)
            .add_edge("__start__", "chat_with_context")
            .compile(name="Qwen Chat Agent")
        ) """         
