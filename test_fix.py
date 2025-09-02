#!/usr/bin/env python3
"""测试消息格式转换修复。"""

import asyncio
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.graph import State, chat_with_context
from langchain_core.messages import HumanMessage, AIMessage


async def test_message_conversion():
    """测试消息格式转换。"""
    print("🧪 测试消息格式转换...")
    
    # 测试1: 字典格式的消息（你遇到的情况）
    print("\n1. 测试字典格式消息...")
    dict_message = {
        'id': '014712db-4943-44c9-ad42-92c773ca05b0', 
        'type': 'human', 
        'content': [{'type': 'text', 'text': '22'}]
    }
    
    state = State(messages=[dict_message])
    
    # 模拟运行时上下文
    class MockRuntime:
        def __init__(self):
            self.context = None
    
    runtime = MockRuntime()
    
    try:
        result = await chat_with_context(state, runtime)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试2: 标准LangChain消息对象
    print("\n2. 测试标准LangChain消息...")
    standard_message = HumanMessage(content="你好")
    state2 = State(messages=[standard_message])
    
    try:
        result2 = await chat_with_context(state2, runtime)
        print(f"结果: {result2}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(test_message_conversion())
