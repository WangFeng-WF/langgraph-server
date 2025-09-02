"""Configuration management for the LangGraph agent."""

import os
from typing import Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """Configuration class for the agent."""
    
    # 通义千问API密钥
    DASHSCOPE_API_KEY: str = os.getenv(
        "DASHSCOPE_API_KEY", 
        "sk-7c61b5435ea94666b3a50d4a0d889bd2"
    )
    
    # 模型配置
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-plus")
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8123"))
    
    # 调试模式
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # 存储配置
    #USE_LOCAL_STORAGE: bool = os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true"
    #DB_URI: str = os.getenv("DB_URI", "postgresql://postgres:kotldb@64@192.168.58.64:5432/langgraph?sslmode=disable")
    
    @classmethod
    def get_api_key(cls) -> str:
        """获取API密钥，优先使用环境变量中的密钥。"""
        return cls.DASHSCOPE_API_KEY
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效。"""
        if not cls.DASHSCOPE_API_KEY:
            print("警告: 未设置DASHSCOPE_API_KEY环境变量")
            return False
        return True
