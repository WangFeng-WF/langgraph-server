#!/usr/bin/env python3
"""测试safe_sql_execute工具的脚本"""

import asyncio
import sys
import os
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tools.safe_sql_execute import safe_sql_execute, safe_sql_execute_tool


def test_safe_sql_execute_function():
    """测试safe_sql_execute函数"""
    print("🧪 测试safe_sql_execute函数...")
    
    # 测试1: 安全的SELECT查询
    print("\n1. 测试安全的SELECT查询...")
    result = safe_sql_execute(
        sql="SELECT 1 as test",
        host="localhost",
        port=3306,
        user="root",
        password="",
        database=""
    )
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 测试2: 危险的DROP操作
    print("\n2. 测试危险的DROP操作...")
    result = safe_sql_execute(
        sql="DROP TABLE test_table",
        host="localhost",
        port=3306,
        user="root",
        password="",
        database=""
    )
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 测试3: 无效的SQL语法
    print("\n3. 测试无效的SQL语法...")
    result = safe_sql_execute(
        sql="INVALID SQL",
        host="localhost",
        port=3306,
        user="root",
        password="",
        database=""
    )
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


def test_safe_sql_execute_tool():
    """测试safe_sql_execute_tool工具"""
    print("\n🧪 测试safe_sql_execute_tool工具...")
    
    # 测试1: 使用工具执行安全查询
    print("\n1. 测试工具执行安全查询...")
    tool_input = json.dumps({
        "sql": "SELECT 1 as test",
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": ""
    })
    
    result = safe_sql_execute_tool._run(tool_input)
    print(f"工具结果: {result}")
    
    # 测试2: 使用工具执行危险操作
    print("\n2. 测试工具执行危险操作...")
    tool_input = json.dumps({
        "sql": "DELETE FROM users",
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": ""
    })
    
    result = safe_sql_execute_tool._run(tool_input)
    print(f"工具结果: {result}")


async def test_agent_with_tools():
    """测试带有工具的代理"""
    print("\n🧪 测试带有工具的代理...")
    
    try:
        from agent.graph import create_agent_with_tools
        
        # 创建代理
        agent_executor = create_agent_with_tools()
        
        # 测试代理
        print("\n1. 测试代理执行简单查询...")
        result = await agent_executor.ainvoke({
            "input": "请执行SQL查询：SELECT 1 as test",
            "chat_history": []
        })
        print(f"代理结果: {result}")
        
        # 测试代理执行危险操作
        print("\n2. 测试代理执行危险操作...")
        result = await agent_executor.ainvoke({
            "input": "请执行SQL查询：DROP TABLE users",
            "chat_history": []
        })
        print(f"代理结果: {result}")
        
    except Exception as e:
        print(f"代理测试失败: {e}")


if __name__ == "__main__":
    print("🚀 开始测试safe_sql_execute工具...")
    
    # 测试函数
    test_safe_sql_execute_function()
    
    # 测试工具
    test_safe_sql_execute_tool()
    
    # 测试代理
    asyncio.run(test_agent_with_tools())
    
    print("\n✅ 所有测试完成！")
