#!/usr/bin/env python3
"""API测试脚本 for LangGraph Qwen Chat Agent."""

import requests
import json
import time


def test_api():
    """测试API接口。"""
    base_url = "http://localhost:8123"
    
    print("🧪 开始测试API接口...")
    
    # 测试1: 健康检查
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试2: 配置信息
    print("\n2. 测试配置信息...")
    try:
        response = requests.get(f"{base_url}/config")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试3: 聊天接口
    print("\n3. 测试聊天接口...")
    try:
        data = {
            "message": "你好，请介绍一下自己",
            "history": [],
            "system_prompt": "你是一个友好的AI助手"
        }
        
        response = requests.post(
            f"{base_url}/chat",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"AI回复: {result.get('response', '无回复')}")
        print(f"对话历史长度: {len(result.get('history', []))}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试4: 上下文对话
    print("\n4. 测试上下文对话...")
    try:
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是基于通义千问的AI助手。"}
        ]
        
        data = {
            "message": "我刚才说了什么？",
            "history": history
        }
        
        response = requests.post(
            f"{base_url}/chat",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"AI回复: {result.get('response', '无回复')}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试5: 重置对话
    print("\n5. 测试重置对话...")
    try:
        response = requests.post(f"{base_url}/reset")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n✅ API测试完成！")


if __name__ == "__main__":
    test_api()
