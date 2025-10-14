# -*- coding: utf-8 -*-
# dbt_react_agent_bak.py

import os
import subprocess
import sys
import uuid
import json
import yaml
import requests
import logging
import asyncio
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from src.tools.dashscope import ChatDashscope
from src.tools.tableInfoQuery import get_table_and_field_info2
from src.tools.dbt_prompt_templates import (
    PROMPT_HEADER, PROMPT_RULES_DBT, PROMPT_RULES_CRON,
    PROMPT_WORKFLOW, PROMPT_CONTEXT_HOLDER, PROMPT_FOOTER
)

# ==============================================================================
# 0. 日志与环境配置
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
load_dotenv()


def setup_encoding():
    try:
        if os.name == 'nt': os.system('chcp 65001 > nul')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.error(f"编码设置失败: {e}")


setup_encoding()


# ==============================================================================
# 1. Agent配置
# ==============================================================================
class DbtAgentConfig:
    LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ca949c46e4904479927923a41562d4d3")
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME = "qwen-max"
    DBT_PROJECT_PATH = os.path.abspath(os.getenv("DBT_PROJECT_PATH", "./src/dbt-core-agent/dbt-core-project"))
    DBT_MODELS_PATH = os.path.join(DBT_PROJECT_PATH, "models")
    DBT_MANIFEST_PATH = os.path.join(DBT_PROJECT_PATH, "target", "manifest.json")
    SCHEDULER_API_ENDPOINT = os.getenv("SCHEDULER_API_ENDPOINT",
                                       "http://192.168.47.24:30027/admin-api/infra/job/create")
    SCHEDULER_API_TOKEN = os.getenv("SCHEDULER_API_TOKEN")


# ==============================================================================
# 2. 核心系统提示词
# ==============================================================================
ASSEMBLED_PROMPT_TEMPLATE = f"""
    {PROMPT_HEADER}

    {PROMPT_RULES_DBT}

    {PROMPT_RULES_CRON}

    {PROMPT_WORKFLOW}

    {PROMPT_CONTEXT_HOLDER}

    {PROMPT_FOOTER}
    """


# ==============================================================================
# 3. 工具定义
# ==============================================================================

class GetUpstreamColumnDescriptionsArgs(BaseModel):
    model_name: str = Field(description="需要查找其上游字段描述的dbt模型的名称。")

# 缓存 manifest.json 内容以提高性能
MANIFEST_CACHE = {"mtime": 0, "content": None}

def get_manifest_content() -> dict:
    """
    一个带缓存的辅助函数,用于高效地读取和解析 manifest.json。
    只有当文件被修改后才会重新从磁盘读取。
    """
    try:
        manifest_path = DbtAgentConfig.DBT_MANIFEST_PATH
        # 如果文件不存在,直接返回空
        if not os.path.exists(manifest_path):
            return {}

        mtime = os.path.getmtime(manifest_path)
        # 如果文件未被修改,且缓存中已有内容,直接返回缓存
        if mtime == MANIFEST_CACHE["mtime"] and MANIFEST_CACHE["content"]:
            return MANIFEST_CACHE["content"]

        # 否则,从磁盘读取文件并更新缓存
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        MANIFEST_CACHE["mtime"] = mtime
        MANIFEST_CACHE["content"] = manifest
        logger.info("成功加载并缓存了 manifest.json。")
        return manifest
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.warning(f"读取或解析 manifest.json 时出现问题: {e}。将返回空内容。")
        # 清空缓存以防内容损坏
        MANIFEST_CACHE["content"] = None
        return {}

@tool(args_schema=GetUpstreamColumnDescriptionsArgs)
def get_upstream_column_descriptions(model_name: str) -> dict:
    """
    分析dbt的manifest.json文件,找到指定模型的上游依赖(ref),并返回其字段(column)的描述。
    这用于在向用户提问前,智能地预填充已知字段的描述。
    """
    logger.info(f"\n--- 正在调用工具 [get_upstream_column_descriptions] for model '{model_name}' ---")
    manifest = get_manifest_content()
    if not manifest:
        return {"status": "Success", "descriptions": {}, "message": "manifest.json 不可用,无法继承描述。"}

    nodes = manifest.get("nodes", {})
    target_node_id = None
    # 查找当前模型的完整节点ID
    for node_id, node_info in nodes.items():
        if node_info.get("resource_type") == "model" and node_info.get("name") == model_name:
            target_node_id = node_id
            break

    if not target_node_id:
        return {"status": "Success", "descriptions": {}, "message": f"在 manifest.json 中未找到模型 '{model_name}'。"}

    upstream_descriptions = {}
    dependency_nodes = nodes[target_node_id].get("depends_on", {}).get("nodes", [])

    for dep_id in dependency_nodes:
        # 我们只关心来自其他dbt模型(ref)的依赖
        if dep_id.startswith("model."):
            upstream_node = nodes.get(dep_id, {})
            upstream_columns = upstream_node.get("columns", {})
            for col_name, col_info in upstream_columns.items():
                description = col_info.get("description")
                # 只继承非空的描述,且不覆盖已有的(虽然此处逻辑简单,可扩展)
                if description and col_name not in upstream_descriptions:
                    upstream_descriptions[col_name] = description

    if not upstream_descriptions:
        return {"status": "Success", "descriptions": {}, "message": "未找到可继承的上游字段描述。"}

    logger.info(f"为模型 '{model_name}' 成功继承了 {len(upstream_descriptions)} 个字段的描述。")
    return {"status": "Success", "descriptions": upstream_descriptions}

class CreateDbtModelFilesArgs(BaseModel):
    model_name: str = Field(description="dbt模型的文件名，不含.sql后缀，例如 'fct_daily_orders'。")
    sql_content: str = Field(description="经过用户确认的完整SELECT查询语句。")
    model_description: str = Field(description="对该模型的业务用途的中文描述。")
    column_descriptions: Dict[str, str] = Field(description="一个字典，映射字段名到其中文描述。")


class ValidateDbtModelArgs(BaseModel):
    model_name: str = Field(description="需要执行 dbt run 进行验证的模型的名称（文件名，不含后缀）。")


@tool(args_schema=ValidateDbtModelArgs)
def validate_dbt_model(model_name: str) -> dict:
    """
    对一个已创建的 dbt 模型安全地执行 'dbt run' 命令，以验证其 SQL 的正确性。
    此工具只会在数据库中创建或更新模型对应的目标表，不会删除任何其他数据。
    这是部署前最关键的安全验证步骤。
    """
    logger.info(f"\n--- 正在调用工具 [validate_dbt_model] for model '{model_name}' ---")
    if not os.path.exists(DbtAgentConfig.DBT_PROJECT_PATH):
        return {"status": "Failed", "message": f"dbt 项目路径不存在: {DbtAgentConfig.DBT_PROJECT_PATH}"}

    try:
        # 使用安全的 'dbt run' 命令，并且不带 '--full-refresh'
        command = ["dbt", "run", "--select", model_name]

        # 1. 创建一个环境变量副本，并强制设置Python子进程的IO编码为UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # 2. 在调用subprocess.run时传入env和encoding参数
        process = subprocess.run(
            command,
            cwd=DbtAgentConfig.DBT_PROJECT_PATH,
            capture_output=True,
            check=False,
            encoding='utf-8',  # 告知subprocess以UTF-8解码
            errors='replace',    # 解码出错时的策略
            env=env            # 传入修改后的环境变量
        )

        # 3. 直接使用process.stdout，它现在已经是解码后的字符串了
        stdout_str = process.stdout
        stderr_str = process.stderr

        if process.returncode == 0:
            logger.info(f"'dbt run' for model '{model_name}' 成功。")
            return {"status": "Success", "message": f"模型 '{model_name}' 的SQL验证通过。", "output": stdout_str}
        else:
            logger.error(f"'dbt run' for model '{model_name}' 失败。")
            error_details = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}"
            return {"status": "Failed", "message": f"模型 '{model_name}' 的SQL验证失败。",
                    "error_details": error_details}

    except FileNotFoundError:
        logger.error("'dbt' command not found. 请确保 dbt-core 已安装并且在系统 PATH 中。")
        return {"status": "Failed", "message": "'dbt' command not found. 请确保 dbt-core 已安装并且在系统 PATH 中。"}
    except Exception as e:
        logger.error(f"执行 dbt run 时发生未知错误: {e}", exc_info=True)
        return {"status": "Failed", "message": f"执行 dbt run 时发生未知错误: {e}"}


# --- [新增] 数据预览工具 ---
class PreviewDbtModelDataArgs(BaseModel):
    model_name: str = Field(description="需要使用 dbt show 预览数据的模型的名称。")


def _format_dbt_show_as_markdown(text_output: str) -> str:
    """一个辅助函数,将dbt show的文本输出转换为Markdown表格。"""
    lines = text_output.strip().split('\n')

    # 如果输出行数太少,可能不是一个有效的表格,直接返回原始文本
    if len(lines) < 3:
        return f"```\n{text_output}\n```" # 使用代码块包裹

    # dbt show的输出通常是: Header, Separator, Data...
    # 我们需要找到分隔符行,它通常包含 '---'
    separator_index = -1
    for i, line in enumerate(lines):
        if '---' in line and '|' in line:
            separator_index = i
            break

    if separator_index == -1:
        return f"```\n{text_output}\n```"

    header = lines[separator_index - 1]
    # 根据表头重新生成标准的Markdown分隔符
    num_columns = header.count('|') - 1
    if num_columns <= 0:
        return f"```\n{text_output}\n```"
    markdown_separator = '| ' + ' | '.join(['---'] * num_columns) + ' |'

    # 替换原始分隔符
    lines[separator_index] = markdown_separator

    return '\n'.join(lines)


@tool(args_schema=PreviewDbtModelDataArgs)
def preview_dbt_model_data(model_name: str) -> dict:
    """
    对一个已验证的模型执行 'dbt show' 命令,以获取前10条数据样本供用户进行业务确认。
    这是一个只读、安全的操作,是调度前的最后一步业务逻辑验证。
    """
    logger.info(f"\n--- 正在调用工具 [preview_dbt_model_data] for model '{model_name}' ---")
    # ... (try/except 和 subprocess.run 的前半部分代码保持不变) ...
    try:
        command = ["dbt", "show", "--select", model_name, "--limit", "10"]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        process = subprocess.run(
            command,
            cwd=DbtAgentConfig.DBT_PROJECT_PATH,
            capture_output=True,
            check=False,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        stdout_str = process.stdout
        stderr_str = process.stderr

        if process.returncode == 0:
            logger.info(f"'dbt show' for model '{model_name}' 成功。")

            # [修改] 将输出转换为Markdown表格
            markdown_preview = _format_dbt_show_as_markdown(stdout_str)

            return {"status": "Success", "message": "数据预览生成成功。", "data_preview": markdown_preview}
        else:
            logger.error(f"'dbt show' for model '{model_name}' 失败。")
            error_details = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}"
            return {"status": "Failed", "message": "生成数据预览失败。", "error_details": error_details}

    except FileNotFoundError:
        # ... (异常处理部分代码保持不变) ...
        return {"status": "Failed", "message": "'dbt' command not found."}
    except Exception as e:
        # ... (异常处理部分代码保持不变) ...
        return {"status": "Failed", "message": f"执行 dbt show 时发生未知错误: {e}"}

# get_table_and_field_info2
@tool
def get_table_and_field_info(prompt_info: str) -> dict:
    """获取指定模型的表和字段信息。"""
    logger.info(f"\n--- 正在调用工具 [get_table_and_field_info] for prompt_info： '{prompt_info}' ---")
    # 这里可以添加实际的实现逻辑
    try:
        table_info = get_table_and_field_info2(prompt_info)
        return {"status": "Success", "table_info": table_info}
    except Exception as e:
        logger.error(f"获取表和字段信息时出错: {e}", exc_info=True)
        return {"status": "Failed", "message": str(e)}


@tool(args_schema=CreateDbtModelFilesArgs)
def create_dbt_model_files(model_name: str, sql_content: str, model_description: str,
                           column_descriptions: Dict[str, str]) -> dict:
    """在dbt项目中创建或更新模型的.sql文件和对应的schema.yml文件。这是dbt-core-agent的核心实现。"""
    logger.info(f"\n--- 正在调用工具 [create_dbt_model_files] for model '{model_name}' ---")
    try:
        sql_file_path = os.path.join(DbtAgentConfig.DBT_MODELS_PATH, f"{model_name}.sql")
        yml_file_path = os.path.join(DbtAgentConfig.DBT_MODELS_PATH, f"{model_name}_schema.yml")
        os.makedirs(DbtAgentConfig.DBT_MODELS_PATH, exist_ok=True)

        # [修改] 在写入文件前，清理SQL内容中可能存在的前后空格和末尾的分号
        cleaned_sql_content = sql_content.strip().rstrip(';')

        sql_file_content = f"{{{{ config(materialized='table') }}}}\n\n{cleaned_sql_content}"

        with open(sql_file_path, "w", encoding="utf-8") as f:
            f.write(sql_file_content)
        logger.info(f"成功写入SQL文件: {sql_file_path}")
        yml_data = {
            "version": 2,
            "models": [{"name": model_name, "description": model_description,
                        "columns": [{"name": name, "description": desc} for name, desc in column_descriptions.items()]}]
        }
        with open(yml_file_path, "w", encoding="utf-8") as f:
            yaml.dump(yml_data, f, allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"成功写入YML文件: {yml_file_path}")
        return {"status": "Success", "message": f"模型 '{model_name}' 的文件已成功创建。"}
    except Exception as e:
        logger.error(f"创建dbt模型文件时出错: {e}", exc_info=True)
        return {"status": "Failed", "message": f"创建文件时发生错误: {e}"}


class ScheduleDbtRunTaskArgs(BaseModel):
    task_name: str = Field(description="定时任务的显示名称。")
    dbt_model_name: str = Field(description="需要定时运行的dbt模型的名称（文件名，不含后缀）。")
    cron_expression: str = Field(description="任务执行时间的Cron表达式。")


@tool(args_schema=ScheduleDbtRunTaskArgs)
def schedule_dbt_run_task(task_name: str, dbt_model_name: str, cron_expression: str) -> dict:
    """调用外部API来为dbt模型创建一个定时调度任务。"""
    logger.info(f"\n--- 正在调用工具 [schedule_dbt_run_task] for model '{dbt_model_name}' ---")
    if not DbtAgentConfig.SCHEDULER_API_ENDPOINT:
        logger.warning("SCHEDULER_API_ENDPOINT 未配置，跳过API调用。")
        return {"status": "Skipped", "message": "定时任务API未配置，跳过创建。"}
    try:
        # 1. 获取token
        login_url = "http://192.168.47.24:30027/admin-api/system/auth/login"
        login_payload = {
            "tenantName": "芋道源码",
            "username": "bossai",
            "password": "123456"
        }
        login_headers = {"Content-Type": "application/json"}
        login_resp = requests.post(login_url, json=login_payload, headers=login_headers, timeout=10)
        login_resp.raise_for_status()
        login_data = login_resp.json()
        access_token = login_data.get("data", {}).get("accessToken")
        if not access_token:
            logger.error(f"登录API未返回accessToken: {login_data}")
            return {"status": "Failed", "message": "登录API未返回accessToken"}

        # 2. 调用调度API
        payload = {"name": task_name, "handlerName": "procedureSyncJob", "handlerParam": dbt_model_name,
                   "cronExpression": cron_expression, "retryCount": "0", "retryInterval": "0"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
        response = requests.post(DbtAgentConfig.SCHEDULER_API_ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"API调用成功，响应: {response.json()}")

        # 解析response.text: '{"code":1001001005,"data":null,"msg":"CRON 表达式不正确"}'
        # 当code 为0的时候表示成功，其他值表示失败
        resp_data = response.json()
        if resp_data.get("code") != 0:
            error_msg = resp_data.get("msg", "未知错误")
            logger.error(f"调度API返回错误: {resp_data}")
            return {"status": "Failed", "message": f"调度API错误: {error_msg}"}

        return {"status": "Success", "message": "定时任务已成功创建。"}
    except requests.exceptions.RequestException as e:
        logger.error(f"调用调度API时出错: {e}", exc_info=True)
        return {"status": "Failed", "message": f"调用API失败: {e}"}

class ProposeUniqueModelNameArgs(BaseModel):
    base_name: str = Field(description="建议的模型基础名称，例如 'fct_orders'。")

def get_existing_models() -> List[str]:
    """一个辅助函数，用于从manifest中获取已存在的模型列表。"""
    # [修改] 调用新的缓存函数
    manifest = get_manifest_content()
    model_nodes = [
        node_name.split('.')[-1]
        for node_name, node_info in manifest.get('nodes', {}).items()
        if node_info.get('resource_type') == 'model'
    ]
    return sorted(list(set(model_nodes)))

@tool(args_schema=ProposeUniqueModelNameArgs)
def propose_unique_model_name(base_name: str) -> dict:
    """
    接收一个建议的模型基础名称，检查其是否存在。如果存在，则自动添加版本后缀 (_v2, _v3, ...) 直到找到一个唯一的名称为止。
    这是在向用户提议模型名称之前的最后一步，以确保不会发生冲突。
    """
    logger.info(f"\n--- 正在调用工具 [propose_unique_model_name] for base_name '{base_name}' ---")
    try:
        existing_models = get_existing_models()
        existing_models_set = set(existing_models)

        if base_name not in existing_models_set:
            logger.info(f"基础名称 '{base_name}' 是唯一的。")
            return {"status": "Success", "unique_name": base_name}

        # 如果基础名称已存在，开始寻找新版本
        version = 2
        while True:
            new_name = f"{base_name}_v{version}"
            if new_name not in existing_models_set:
                logger.info(f"找到唯一名称 '{new_name}'。")
                return {"status": "Success", "unique_name": new_name}
            version += 1

    except Exception as e:
        logger.error(f"检查模型名称唯一性时出错: {e}", exc_info=True)
        return {"status": "Failed", "message": str(e)}



tools = [
    get_table_and_field_info,
    create_dbt_model_files,
    validate_dbt_model,
    preview_dbt_model_data,
    schedule_dbt_run_task,
    propose_unique_model_name,
    get_upstream_column_descriptions
]

# ==============================================================================
# 4. LLM 初始化
# ==============================================================================
try:
    if not DbtAgentConfig.LLM_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置。")
    llm = ChatDashscope(model=DbtAgentConfig.LLM_MODEL_NAME, api_key=DbtAgentConfig.LLM_API_KEY,
                        base_url=DbtAgentConfig.LLM_BASE_URL)
    logger.info(f"模型 '{DbtAgentConfig.LLM_MODEL_NAME}' 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败: {e}", exc_info=True)
    sys.exit(1)


# ==============================================================================
# 5. Agent 构建 (核心改动)
# ==============================================================================

def get_simplified_dbt_context(manifest_path: str) -> str:
    """从manifest.json中提取简化的、对LLM友好的上下文。"""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        nodes = manifest.get("nodes", {})
        sources = manifest.get("sources", {})
        model_names = [key.split('.')[-1] for key, value in nodes.items() if value['resource_type'] == 'model']
        # source_tuples = [(key.split('.')[1], key.split('.')[2]) for key in sources.keys()]
        source_tuples = [
            (value['source_name'], value['name'])
            for key, value in sources.items()
            if 'source_name' in value and 'name' in value
        ]

        context = {"models": sorted(list(set(model_names))),
                   "sources": [{"source_name": s[0], "table_name": s[1]} for s in sorted(list(set(source_tuples)))]}
        return json.dumps(context, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        logger.warning(f"{manifest_path} 未找到，dbt上下文将为空。")
        return "{}"
    except Exception as e:
        logger.error(f"解析 manifest.json 时出错: {e}")
        return "{}"


# --- [修复] 关键部分：使用新的 API ---

# 打印dbt_context目录
logger.info(f"尝试加载 dbt 上下文文件: {DbtAgentConfig.DBT_MANIFEST_PATH}")

# 1. 加载一次 dbt 上下文
dbt_context = get_simplified_dbt_context(DbtAgentConfig.DBT_MANIFEST_PATH)

logger.info(f"提取的 dbt 上下文: {dbt_context}")

# 2. 直接格式化最终的系统提示字符串
final_system_prompt = ASSEMBLED_PROMPT_TEMPLATE.format(dbt_context=dbt_context)

# 3. 使用 create_react_agent 构建 Agent，并直接传递 system_message 参数
#    我们不再需要 create_system_message_modifier 函数了。
analysis_agent_graph = create_react_agent(
    llm,
    tools,
    prompt=final_system_prompt,  # <-- 使用新的 `system_message` 参数
    # checkpointer=MemorySaver()
)

logger.info("✅ dbt_react_agent 图已编译并赋值给 'dbt_agent_graph' 变量。")


# ==============================================================================
# 6. 主函数 (用于本地独立测试)
# ==============================================================================

async def main():
    """
    此函数仅用于直接运行此文件进行本地测试。
    当通过 LangServe 和 langgraph.json 部署时，此部分代码不会被执行。
    """
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    logger.info("--- ReAct Agent 本地测试启动 (输入 'exit' 退出) ---")

    initial_inputs = {"messages": [HumanMessage(content="你好")]}
    async for event in analysis_agent_graph.astream(initial_inputs, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            print(f"\n助手> {last_message.content}")

    while True:
        try:
            user_input = input(f"\n你> ")
            if user_input.lower() in ["exit", "退出"]:
                break
            inputs = {"messages": [HumanMessage(content=user_input)]}
            async for event in dbt_agent_graph.astream(inputs, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    print(f"\n助手> {last_message.content}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"程序发生错误: {e}", exc_info=True)
            break
    logger.info("\n再见！")


if __name__ == "__main__":
    #  ./dbt-core-project\target/manifest.json 未找到，dbt上下文将为空
    if not os.path.exists(DbtAgentConfig.DBT_MANIFEST_PATH):
        logger.warning(f"警告: {DbtAgentConfig.DBT_MANIFEST_PATH} 未找到。")
        logger.warning("请在dbt项目目录下运行 'dbt compile' 以生成该文件。")
        logger.warning("Agent将以无dbt上下文模式运行。")
    asyncio.run(main())
