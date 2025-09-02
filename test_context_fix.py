#!/usr/bin/env python3
"""测试对话上下文保持功能。"""

import asyncio
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.chat_handler import create_chat_handler


async def test_conversation_context():
    """测试对话上下文保持。"""
    print("🧪 测试对话上下文保持功能...")
    
    # 创建聊天处理器
    chat_handler = create_chat_handler()
    
    # 第一轮对话
    print("\n1. 第一轮对话...")
    result1 = await chat_handler.process_message("你好，我叫小明")
    print(f"AI回复: {result1['response']}")
    print(f"对话历史长度: {len(result1['conversation_history'])}")
    
    # 第二轮对话（不传入历史记录，应该保持上下文）
    print("\n2. 第二轮对话（测试上下文保持）...")
    result2 = await chat_handler.process_message("你还记得我的名字吗？")
    print(f"AI回复: {result2['response']}")
    print(f"对话历史长度: {len(result2['conversation_history'])}")
    
    # 第三轮对话
    print("\n3. 第三轮对话...")
    result3 = await chat_handler.process_message("我今天心情很好")
    print(f"AI回复: {result3['response']}")
    print(f"对话历史长度: {len(result3['conversation_history'])}")
    
    # 测试重置功能
    print("\n4. 测试重置功能...")
    reset_result = await chat_handler.reset_conversation()
    print(f"重置结果: {reset_result['response']}")
    
    # 重置后的对话
    print("\n5. 重置后的对话...")
    result4 = await chat_handler.process_message("这是新的对话")
    print(f"AI回复: {result4['response']}")
    print(f"对话历史长度: {len(result4['conversation_history'])}")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    asyncio.run(test_conversation_context())
