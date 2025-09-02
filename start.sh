#!/bin/bash

# LangGraph Qwen Chat Agent 快速启动脚本

echo "🚀 启动 LangGraph Qwen Chat Agent..."

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python 3.9+"
    exit 1
fi

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  建议在虚拟环境中运行"
    echo "   创建虚拟环境: python -m venv venv"
    echo "   激活虚拟环境: source venv/bin/activate (Linux/Mac) 或 venv\\Scripts\\activate (Windows)"
fi

# 安装依赖
echo "📦 安装依赖..."
pip install -e .

# 检查环境变量
if [[ -z "$DASHSCOPE_API_KEY" ]]; then
    echo "⚠️  未设置DASHSCOPE_API_KEY环境变量"
    echo "   请设置: export DASHSCOPE_API_KEY=your_key"
    echo "   或创建.env文件"
fi

# 启动服务器
echo "🌐 启动服务器..."
python run_server.py
