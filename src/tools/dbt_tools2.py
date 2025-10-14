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
import yaml

# 2. 从环境变量获取数据库凭据 (最佳实践)
#    在运行 Agent 之前，您需要在终端中设置这些环境变量
DB_HOST = "192.168.58.75"
DB_PORT = "5432"
DB_NAME = "kotl_tool"
DB_USER = "postgres"
DB_PASSWORD = "kotl2025"

# --- 配置 ---
# 将此路径更改为您的 dbt 项目的绝对或相对路径
# DBT_PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dbt_project')
DBT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dbt-core-agent/dbt-core-project'))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_compiled_sql_for_model(model_name: str) -> Optional[str]:
    """辅助函数：找到并读取模型编译后的 SQL 文件。"""
    try:
        project_name = os.path.basename(DBT_PROJECT_DIR)
        # 编译后的 SQL 文件路径通常遵循这个模式
        compiled_path = Path(DBT_PROJECT_DIR) / "target/compiled" / project_name / f"models/{model_name}.sql"
        if not compiled_path.exists():
            # 检查是否在 generated 子目录中
            compiled_path = Path(DBT_PROJECT_DIR) / "target/compiled" / project_name / f"models/generated/{model_name}.sql"

        if compiled_path.exists():
            with open(compiled_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logging.error(f"无法读取编译后的 SQL 文件 for {model_name}: {e}")
    return None

def _get_model_raw_sql(model_name: str) -> Optional[str]:
    """辅助函数：获取模型的原始 Jinja SQL 代码。"""
    try:
        model_path = Path(DBT_PROJECT_DIR) / f"models/{model_name}.sql"
        if not model_path.exists():
            model_path = Path(DBT_PROJECT_DIR) / f"models/generated/{model_name}.sql"

        if model_path.exists():
            with open(model_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logging.error(f"无法读取原始 SQL 文件 for {model_name}: {e}")
    return None

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
def run_dbt_model(model_name: str) -> dict:
    """
    运行一个指定的 dbt 模型，并能对错误进行解析、尝试自动修复。
    返回一个包含状态、消息和给LLM建议的字典。
    """
    command = ["dbt", "run", "--select", model_name]
    result = _run_dbt_command(command)

    if result.returncode == 0:
        return {
            "status": "success",
            "message": f"dbt 模型 '{model_name}' 成功运行。",
            "details": result.stdout,
            "suggestion_for_llm": "无。任务成功完成。"
        }

    # --- 如果失败，则进入错误分析和处理流程 ---
    error_log = result.stderr
    full_log = result.stdout + "\n" + error_log

    # 1. 解析错误：依赖问题 (Model not found)
    dependency_error = re.search(r"dbt found a ref to '(\w+)' but the model was not found", error_log)
    if dependency_error:
        missing_model = dependency_error.group(1)
        logging.warning(f"检测到依赖问题: 模型 '{missing_model}' 未找到。尝试自动修复...")

        # 2. 自动修复：尝试运行模型及其所有父模型
        # 使用 dbt run --select +{model_name}
        fix_command = ["dbt", "run", "--select", f"+{model_name}"]
        fix_result = _run_dbt_command(fix_command)

        if fix_result.returncode == 0:
            return {
                "status": "success",
                "message": f"检测到依赖问题并已自动修复。模型 '{model_name}' 及其依赖项已成功运行。",
                "details": fix_result.stdout,
                "suggestion_for_llm": "无。任务成功完成。"
            }
        else:
            # 如果自动修复仍然失败
            error_log = fix_result.stderr

    # 1. 解析错误：SQL 编译或数据库错误
    sql_error = re.search(r"(Compilation Error|Database Error at model)", error_log)
    if sql_error:
        compiled_sql = _get_compiled_sql_for_model(model_name)
        raw_sql = _get_model_raw_sql(model_name)

        # 3. 提供上下文：构建丰富的错误信息给 LLM
        return {
            "status": "error",
            "message": f"模型 '{model_name}' 存在 SQL 错误。",
            "details": error_log,
            "suggestion_for_llm": (
                "这是一个 SQL 错误。请分析以下信息并修复原始 SQL 代码。\n"
                f"1. **错误日志**:\n---\n{error_log}\n---\n"
                f"2. **失败时执行的已编译 SQL**:\n---\n{compiled_sql}\n---\n"
                f"3. **需要你修复的原始 Jinja SQL (`{model_name}.sql`)**:\n---\n{raw_sql}\n---\n"
                "请使用 `generate_dbt_files` 工具提交你修复后的原始 Jinja SQL 代码。"
            )
        }

    # 捕获其他未知错误
    return {
        "status": "error",
        "message": f"运行模型 '{model_name}' 时发生未知错误。",
        "details": full_log,
        "suggestion_for_llm": (
            "发生了一个未知类型的 dbt 错误。请仔细分析以下完整日志，并判断下一步应该做什么。\n"
            f"**完整日志**:\n---\n{full_log}\n---"
        )
    }

# 确保这个辅助函数存在于您的文件中
def _get_dbt_project_name(dbt_project_dir: str) -> Optional[str]:
    """
    通过读取 dbt_project.yml 文件来安全地获取 dbt 项目的名称。
    这是最可靠的方法。
    """
    try:
        with open(Path(dbt_project_dir) / "dbt_project.yml", 'r', encoding='utf-8') as f:
            dbt_config = yaml.safe_load(f)
            return dbt_config.get("name")
    except Exception as e:
        logging.error(f"无法读取或解析 dbt_project.yml: {e}")
        return None

@tool
def run_sql_query(sql: str) -> str:
    """
    在 dbt 的目标数据库 (PostgreSQL) 中执行一个 SQL 查询。
    此工具会正确地解析 dbt 的 `ref()` 宏。
    """
    try:
        # 在 dbt 的 'analyses' 目录中创建临时 SQL 文件
        analysis_dir = Path(DBT_PROJECT_DIR) / "analyses"
        analysis_dir.mkdir(exist_ok=True)
        temp_sql_path = analysis_dir / f"tmp_query_{os.urandom(8).hex()}.sql"

        with open(temp_sql_path, 'w', encoding='utf-8') as f:
            f.write(sql)

        # 使用 dbt compile 来获取最终的 SQL
        # --select 参数可以直接使用文件名
        compile_command = ["dbt", "compile", "--select", temp_sql_path.stem]
        result = _run_dbt_command(compile_command)

        # 编译完成后，立即删除临时文件
        os.remove(temp_sql_path)

        if result.returncode != 0:
            return f"编译 dbt SQL 时出错: {result.stderr}"

        # --- 【最终修复的逻辑】---
        # 1. 从 dbt_project.yml 中可靠地获取项目名称
        project_name = _get_dbt_project_name(DBT_PROJECT_DIR)
        if not project_name:
            return "错误: 无法从 dbt_project.yml 中确定项目名称。"

        # 2. 使用 Path 对象和正确的项目名称，健壮地构造路径
        #    这将生成类似 .../target/compiled/jaffle_shop/analyses/tmp_query_....sql 的路径
        compiled_sql_path = Path(DBT_PROJECT_DIR) / "target" / "compiled" / project_name / "analyses" / f"{temp_sql_path.stem}.sql"
        # --- 【修复结束】---

        if not compiled_sql_path.exists():
            logging.error(f"尝试查找文件但失败了: {compiled_sql_path}")
            return f"错误: 找不到编译后的 SQL 文件。预期的路径是: {compiled_sql_path}"

        with open(compiled_sql_path, 'r', encoding='utf-8') as f:
            processed_sql = f.read()

    except Exception as e:
        # 添加更详细的异常日志
        logging.error(f"dbt 编译过程中发生意外错误: {e}", exc_info=True)
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
