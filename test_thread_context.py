#!/usr/bin/env python3
"""
测试基于 thread_id 的上下文对话功能
"""

import asyncio
import json
from src.agent.graph import graph, State, message_store


async def test_thread_context():
    """测试线程上下文对话功能"""
    
    print("=== 测试基于 thread_id 的上下文对话功能 ===\n")
    
    # 测试线程1
    thread_id_1 = "test_thread_001"
    print(f"开始测试线程: {thread_id_1}")
    
    # 第一次对话
    state1 = State(
        user_id="user_001",
        thread_id=thread_id_1,
        system_prompt="你是一个有用的AI助手，专门帮助用户解决问题。"
    )
    
    # 模拟用户消息
    from langchain_core.messages import HumanMessage
    user_msg1 = HumanMessage(content="你好，请介绍一下你自己")
    state1.add_message(user_msg1)
    
    # 调用图处理
    config = {"configurable": {"thread_id": thread_id_1}}
    result1 = await graph.ainvoke(
        {"messages": state1.messages},
        config=config
    )
    
    print(f"第一次对话结果: {len(result1['messages'])} 条消息")
    print(f"最后一条消息: {result1['messages'][-1].content[:100]}...")
    
    # 第二次对话（应该能记住上下文）
    user_msg2 = HumanMessage(content="我刚才问了你什么？")
    state1.add_message(user_msg2)
    
    result2 = await graph.ainvoke(
        {"messages": state1.messages},
        config=config
    )
    
    print(f"第二次对话结果: {len(result2['messages'])} 条消息")
    print(f"最后一条消息: {result2['messages'][-1].content[:100]}...")
    
    # 测试线程2（独立的对话）
    thread_id_2 = "test_thread_002"
    print(f"\n开始测试新线程: {thread_id_2}")
    
    state2 = State(
        user_id="user_002",
        thread_id=thread_id_2,
        system_prompt="你是一个数学老师，专门教授数学知识。"
    )
    
    user_msg3 = HumanMessage(content="请解释一下什么是微积分")
    state2.add_message(user_msg3)
    
    config2 = {"configurable": {"thread_id": thread_id_2}}
    result3 = await graph.ainvoke(
        {"messages": state2.messages},
        config=config2
    )
    
    print(f"新线程对话结果: {len(result3['messages'])} 条消息")
    print(f"最后一条消息: {result3['messages'][-1].content[:100]}...")
    
    # 验证线程隔离
    print(f"\n=== 验证线程隔离 ===")
    print(f"线程 {thread_id_1} 统计: {state1.get_thread_stats()}")
    print(f"线程 {thread_id_2} 统计: {state2.get_thread_stats()}")
    
    # 测试重新加载线程1
    print(f"\n=== 测试重新加载线程 {thread_id_1} ===")
    state1_reload = State(thread_id=thread_id_1)
    print(f"重新加载后消息数量: {len(state1_reload.messages)}")
    print(f"重新加载后统计: {state1_reload.get_thread_stats()}")
    
    # 显示全局存储信息
    print(f"\n=== 全局存储信息 ===")
    print(f"总线程数: {message_store.get_thread_count()}")
    print(f"线程信息: {message_store.get_thread_info()}")


async def test_thread_management():
    """测试线程管理功能"""
    
    print("\n=== 测试线程管理功能 ===\n")
    
    thread_id = "management_test_001"
    
    # 创建线程并添加消息
    state = State(thread_id=thread_id)
    state.add_message(HumanMessage(content="测试消息1"))
    state.add_message(HumanMessage(content="测试消息2"))
    
    print(f"添加消息后统计: {state.get_thread_stats()}")
    
    # 清除线程历史
    state.clear_thread_history()
    print(f"清除历史后统计: {state.get_thread_stats()}")
    
    # 验证全局存储
    print(f"全局存储中线程数: {message_store.get_thread_count()}")


async def main():
    """主测试函数"""
    try:
        await test_thread_context()
        await test_thread_management()
        print("\n=== 所有测试完成 ===")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
