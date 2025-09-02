#!/usr/bin/env python3
"""测试脚本 for LangGraph Qwen Chat Agent."""

import asyncio
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.chat_handler import create_chat_handler


async def test_chat():
    """测试聊天功能。"""
    print("🧪 开始测试聊天功能...")
    
    # 创建聊天处理器
    chat_handler = create_chat_handler()
    
    # 测试1: 欢迎消息
    print("\n1. 测试欢迎消息...")
    result = await chat_handler.get_welcome_message()
    print(f"结果: {result['response']}")
    
    # 测试2: 简单对话
    print("\n2. 测试简单对话...")
    result = await chat_handler.process_message("你好，请介绍一下自己")
    print(f"用户: 你好，请介绍一下自己")
    print(f"AI: {result['response']}")
    
    # 测试3: 上下文对话
    print("\n3. 测试上下文对话...")
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是基于通义千问的AI助手。"}
    ]
    
    result = await chat_handler.process_message(
        "我刚才说了什么？", 
        conversation_history=history
    )
    print(f"用户: 我刚才说了什么？")
    print(f"AI: {result['response']}")
    
    # 测试4: 重置对话
    print("\n4. 测试重置对话...")
    result = await chat_handler.reset_conversation()
    print(f"结果: {result['response']}")
    
    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(test_chat())
