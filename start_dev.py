#!/usr/bin/env python3
"""开发环境启动脚本 for LangGraph Qwen Chat Agent."""

import sys
import os

# 设置开发环境变量
os.environ["USE_LOCAL_STORAGE"] = "true"
os.environ["DEBUG"] = "true"
os.environ["DASHSCOPE_API_KEY"] = "sk-7c61b5435ea94666b3a50d4a0d889bd2"
os.environ["QWEN_MODEL"] = "qwen-plus"
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = "8123"

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🚀 启动 LangGraph Qwen Chat Agent 开发服务器...")
print("📝 使用内存存储模式")
print("🔧 调试模式已启用")

try:
    from agent.server import run_server
    run_server()
except ImportError:
    print("❌ 未找到 server.py 文件，请检查项目结构")
    print("💡 尝试直接运行: langgraph dev")
    sys.exit(1)
