# tools/dbt_tools.py
# -*- coding: utf-8 -*-
import os
import json
import subprocess
import logging
import re
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool

import psycopg2
import pandas as pd
from psycopg2 import sql

# 2. 从环境变量获取数据库凭据 (最佳实践)
#    在运行 Agent 之前，您需要在终端中设置这些环境变量
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "kotl_tool")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "kotl2025")

# --- 配置 ---
# 将此路径更改为您的 dbt 项目的绝对或相对路径
# DBT_PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dbt_project')
DBT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dbt-core-agent/dbt-core-project'))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def _run_dbt_command(command: List[str]) -> subprocess.CompletedProcess:
    """在 dbt 项目目录中安全地运行 dbt 命令。"""
    logging.info(f"Running dbt command: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=DBT_PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

# --- 工具实现 ---

@tool
def get_dbt_metadata(keywords: Optional[str] = None) -> str:
    """
    获取 dbt 项目的元数据，可以根据关键词进行筛选。
    这是 Agent 探索现有模型的第一步。
    """
    manifest_path = Path(DBT_PROJECT_DIR) / "target/manifest.json"

    # 如果 manifest.json 不存在或过时，先运行 dbt parse
    if not manifest_path.exists():
        logging.info("manifest.json not found, running 'dbt parse'...")
        result = _run_dbt_command(["dbt", "parse"])
        if result.returncode != 0:
            return f"Error running dbt parse: {result.stderr}"

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        output = "dbt 项目元数据:\n"
        nodes = manifest.get("nodes", {})

        found_models = []
        for key, value in nodes.items():
            if value.get("resource_type") == "model":
                model_name = value.get("name")
                description = value.get("description", "No description.")
                raw_code = value.get("raw_code", "")

                if keywords:
                    # 简单关键词匹配
                    if keywords.lower() not in model_name.lower() and \
                            keywords.lower() not in description.lower():
                        continue

                columns_str = "\n".join([f"      - {col}: {meta.get('description', 'N/A')}" for col, meta in value.get('columns', {}).items()])
                found_models.append(
                    f"  - 模型名称: {model_name}\n"
                    f"    描述: {description}\n"
                    f"    列:\n{columns_str if columns_str else '      N/A'}"
                )

        if not found_models:
            return "根据关键词未找到匹配的模型。"

        return output + "\n".join(found_models)

    except Exception as e:
        return f"解析 manifest.json 时出错: {e}"

@tool
def generate_dbt_files(model_name: str, sql: str) -> str:
    """
    在 dbt 项目中创建一个新的 SQL 模型文件。
    这是 Agent 动态生成新分析能力的关键。
    """
    # 安全性检查：确保 model_name 是一个有效的文件名
    if not re.match(r'^[a-zA-Z0-9_]+$', model_name):
        return "错误: 模型名称只能包含字母、数字和下划线。"

    # 将新模型放在 'models/generated/' 目录下以示区分
    target_dir = Path(DBT_PROJECT_DIR) / "models/generated"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{model_name}.sql"

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sql)
        return f"成功创建 dbt 模型文件: {file_path.relative_to(DBT_PROJECT_DIR)}"
    except Exception as e:
        return f"创建文件时出错: {e}"

@tool
def run_dbt_model(model_name: str) -> str:
    """
    运行一个指定的 dbt 模型。可用于运行现有模型或刚刚动态创建的模型。
    """
    # 移除了检查数据库是否存在并运行 'dbt seed' 的逻辑，
    # 因为我们假设目标 PG 数据库是持久化且已托管的。

    command = ["dbt", "run", "--select", model_name]
    result = _run_dbt_command(command)

    if result.returncode == 0:
        return f"dbt 模型 '{model_name}' 成功运行。\n日志:\n{result.stdout}"
    else:
        return f"错误: 运行 dbt 模型 '{model_name}' 失败。\n错误日志:\n{result.stderr}"


@tool
def run_sql_query(sql: str) -> str:
    """
    在 dbt 的目标数据库 (PostgreSQL) 中执行一个 SQL 查询。
    此工具会在执行前正确地解析 dbt 的 `ref()` 宏。
    """
    # 使用 dbt compile 来健壮地解析 ref()
    try:
        # 创建一个临时文件，让 dbt 来编译
        # 将临时文件放在 analyses 目录下是 dbt 的标准做法
        temp_sql_path = Path(DBT_PROJECT_DIR) / f"analyses/tmp_query_{os.urandom(8).hex()}.sql"
        temp_sql_path.parent.mkdir(exist_ok=True)
        with open(temp_sql_path, 'w', encoding='utf-8') as f:
            f.write(sql)

        # 新增：先让 dbt 重新扫描项目
        parse_result = _run_dbt_command(["dbt", "parse"])
        if parse_result.returncode != 0:
            os.remove(temp_sql_path)
            return f"dbt parse 失败: {parse_result.stderr}"

        # 使用 dbt compile 获取最终的 SQL
        compile_command = ["dbt", "compile", "--select", temp_sql_path.stem]
        result = _run_dbt_command(compile_command)

        # 立即清理临时文件
        os.remove(temp_sql_path)

        if result.returncode != 0:
            return f"编译 dbt SQL 时出错: {result.stderr}"

        # 找到编译后的 SQL 文件路径
        # 注意: 这里的路径结构依赖于 dbt 的输出，可能需要根据您的项目名称微调
        project_name = DBT_PROJECT_DIR.split('/')[-1]
        compiled_sql_path = Path(DBT_PROJECT_DIR) / "target/compiled" / project_name / f"analyses/{temp_sql_path.stem}.sql"

        if not compiled_sql_path.exists():
            return "错误: 找不到编译后的 SQL 文件。"

        with open(compiled_sql_path, 'r', encoding='utf-8') as f:
            processed_sql = f.read()

    except Exception as e:
        return f"dbt 编译过程中发生错误: {e}"

    logging.info(f"正在执行编译后的 SQL 查询: {processed_sql}")

    # 连接 PostgreSQL 并执行查询
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        # 使用 pandas 轻松地将 SQL 结果读入 DataFrame
        df = pd.read_sql_query(processed_sql, conn)
        conn.close()

        if df.empty:
            return "查询成功执行，但没有返回任何结果。"

        return f"查询成功执行。\n结果:\n{df.to_markdown(index=False)}"

    except Exception as e:
        return f"在 PostgreSQL 上执行 SQL 查询时发生错误: {e}"
