#!/usr/bin/env python3
"""启动脚本 for LangGraph Qwen Chat Agent."""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent.server import run_server

if __name__ == "__main__":
    print("🚀 启动 LangGraph Qwen Chat Agent 服务器...")
    run_server()
