# -*- coding: utf-8 -*-
# report_agent.py

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
# [修改] 引入新的提示词文件
from src.tools.dbt_prompt_templates import (
    PROMPT_HEADER, PROMPT_RULES_DBT, PROMPT_RULES_CRON,
    PROMPT_WORKFLOW, PROMPT_CONTEXT_HOLDER, PROMPT_FOOTER
)
from src.tools.dashscope import ChatDashscope
from src.tools.tableInfoQuery import get_table_and_field_info2

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
# 1. Agent配置 (底层路径不变)
# ==============================================================================
class ReportAgentConfig:
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
# 3. 工具定义 (名称和描述已修改，实现不变)
# ==============================================================================

class GetUpstreamColumnDescriptionsArgs(BaseModel):
    report_name: str = Field(description="需要查找其上游字段描述的报告的名称。")

MANIFEST_CACHE = {"mtime": 0, "content": None}

def get_manifest_content() -> dict:
    """[内部函数] 带缓存地读取和解析 manifest.json。"""
    try:
        manifest_path = ReportAgentConfig.DBT_MANIFEST_PATH
        if not os.path.exists(manifest_path): return {}
        mtime = os.path.getmtime(manifest_path)
        if mtime == MANIFEST_CACHE["mtime"] and MANIFEST_CACHE["content"]:
            return MANIFEST_CACHE["content"]
        with open(manifest_path, 'r', encoding='utf-8') as f: manifest = json.load(f)
        MANIFEST_CACHE["mtime"] = mtime
        MANIFEST_CACHE["content"] = manifest
        logger.info("成功加载并缓存了系统元数据(manifest.json)。")
        return manifest
    except Exception as e:
        logger.warning(f"读取或解析 manifest.json 时出现问题: {e}。")
        MANIFEST_CACHE["content"] = None
        return {}

@tool(args_schema=GetUpstreamColumnDescriptionsArgs)
def get_upstream_column_descriptions(report_name: str) -> dict:
    """
    分析系统元数据,找到指定报告的上游依赖,并返回其字段的描述。
    用于在向用户提问前,智能地预填充已知字段的描述。
    """
    logger.info(f"\n--- 正在调用工具 [get_upstream_column_descriptions] for report '{report_name}' ---")
    manifest = get_manifest_content()
    if not manifest: return {"status": "Success", "descriptions": {}, "message": "系统元数据不可用,无法继承描述。"}
    nodes = manifest.get("nodes", {})
    target_node_id = next((node_id for node_id, node_info in nodes.items() if node_info.get("resource_type") == "model" and node_info.get("name") == report_name), None)
    if not target_node_id: return {"status": "Success", "descriptions": {}, "message": f"在系统元数据中未找到报告 '{report_name}'。"}

    upstream_descriptions = {}
    dependency_nodes = nodes[target_node_id].get("depends_on", {}).get("nodes", [])
    for dep_id in dependency_nodes:
        if dep_id.startswith("model."):
            upstream_node = nodes.get(dep_id, {})
            upstream_columns = upstream_node.get("columns", {})
            for col_name, col_info in upstream_columns.items():
                description = col_info.get("description")
                if description and col_name not in upstream_descriptions:
                    upstream_descriptions[col_name] = description
    return {"status": "Success", "descriptions": upstream_descriptions}

class CreateReportFilesArgs(BaseModel):
    report_name: str = Field(description="报告的文件名，不含.sql后缀，例如 'fct_daily_orders'。")
    sql_content: str = Field(description="经过用户确认的完整SELECT查询语句。")
    model_description: str = Field(description="对该报告的业务用途的中文描述。")
    column_descriptions: Dict[str, str] = Field(description="一个字典，映射字段名到其中文描述。")

# [修改] 工具名和描述
@tool(args_schema=CreateReportFilesArgs)
def create_report_files(report_name: str, sql_content: str, model_description: str,
                        column_descriptions: Dict[str, str]) -> dict:
    """在后台项目中创建或更新报告的.sql文件和对应的schema.yml文件。"""
    logger.info(f"\n--- 正在调用工具 [create_report_files] for report '{report_name}' ---")
    try:
        # [不变] 底层文件路径和dbt格式完全不变
        sql_file_path = os.path.join(ReportAgentConfig.DBT_MODELS_PATH, f"{report_name}.sql")
        yml_file_path = os.path.join(ReportAgentConfig.DBT_MODELS_PATH, f"{report_name}_schema.yml")
        os.makedirs(ReportAgentConfig.DBT_MODELS_PATH, exist_ok=True)

        cleaned_sql_content = sql_content.strip().rstrip(';')
        sql_file_content = f"{{{{ config(materialized='table') }}}}\n\n{cleaned_sql_content}"

        with open(sql_file_path, "w", encoding="utf-8") as f: f.write(sql_file_content)
        logger.info(f"成功写入SQL文件: {sql_file_path}")

        yml_data = {
            "version": 2,
            "models": [{"name": report_name, "description": model_description,
                        "columns": [{"name": name, "description": desc} for name, desc in column_descriptions.items()]}]
        }
        with open(yml_file_path, "w", encoding="utf-8") as f: yaml.dump(yml_data, f, allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"成功写入YML文件: {yml_file_path}")

        return {"status": "Success", "message": f"报告 '{report_name}' 的文件已成功创建。"}
    except Exception as e:
        logger.error(f"创建报告文件时出错: {e}", exc_info=True)
        return {"status": "Failed", "message": f"创建文件时发生错误: {e}"}

class ValidateReportLogicArgs(BaseModel):
    report_name: str = Field(description="需要执行后台验证的报告的名称。")

# [修改] 工具名和描述
@tool(args_schema=ValidateReportLogicArgs)
def validate_report_logic(report_name: str) -> dict:
    """对一个已创建的报告安全地执行其SQL逻辑，以验证其正确性。"""
    logger.info(f"\n--- 正在调用工具 [validate_report_logic] for report '{report_name}' ---")
    # [不变] 底层dbt命令完全不变
    try:
        command = ["dbt", "run", "--select", report_name]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        process = subprocess.run(
            command, cwd=ReportAgentConfig.DBT_PROJECT_PATH, capture_output=True, check=False,
            encoding='utf-8', errors='replace', env=env
        )
        if process.returncode == 0:
            clean_output = _filter_dbt_output(process.stdout)
            return {"status": "Success", "message": f"报告 '{report_name}' 的SQL验证通过。", "output": clean_output}
        else:
            error_details = f"STDOUT:\n{_filter_dbt_output(process.stdout)}\n\nSTDERR:\n{process.stderr}"
            return {"status": "Failed", "message": f"报告 '{report_name}' 的SQL验证失败。", "error_details": error_details}
    except Exception as e:
        return {"status": "Failed", "message": f"执行SQL验证时发生未知错误: {e}"}

def _filter_dbt_output(output: str) -> str:
    """过滤掉包含 dbt 相关信息的行，仅保留核心输出。"""
    lines = output.splitlines()
    filtered = [line for line in lines if not any(
        kw in line for kw in [
            "dbt=", "Registered adapter", "Concurrency:", "Found ", "Finished running",
            "Completed successfully", "Done. PASS=", "START sql table model", "OK created sql table model"
        ]
    )]
    return "\n".join(filtered).strip()

class PreviewReportDataArgs(BaseModel):
    report_name: str = Field(description="需要使用后台命令预览数据的报告的名称。")

def _format_dbt_show_as_markdown(text_output: str) -> str:
    # [不变] 内部辅助函数不变
    lines = text_output.strip().split('\n')
    if len(lines) < 3: return f"```\n{text_output}\n```"
    separator_index = next((i for i, line in enumerate(lines) if '---' in line and '|' in line), -1)
    if separator_index == -1: return f"```\n{text_output}\n```"
    header = lines[separator_index - 1]
    num_columns = header.count('|') - 1
    if num_columns <= 0: return f"```\n{text_output}\n```"
    lines[separator_index] = '| ' + ' | '.join(['---'] * num_columns) + ' |'
    return '\n'.join(lines)

# [修改] 工具名和描述
@tool(args_schema=PreviewReportDataArgs)
def preview_report_data(report_name: str) -> dict:
    """对一个已验证的报告执行后台命令,以获取少量数据样本供用户进行业务确认。"""
    logger.info(f"\n--- 正在调用工具 [preview_report_data] for report '{report_name}' ---")
    # [不变] 底层dbt命令完全不变
    try:
        command = ["dbt", "show", "--select", report_name, "--limit", "10"]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        process = subprocess.run(
            command, cwd=ReportAgentConfig.DBT_PROJECT_PATH, capture_output=True, check=False,
            encoding='utf-8', errors='replace', env=env
        )
        if process.returncode == 0:
            clean_stdout = _filter_dbt_output(process.stdout)
            markdown_preview = _format_dbt_show_as_markdown(clean_stdout)
            return {"status": "Success", "message": "数据预览生成成功。", "data_preview": markdown_preview}
        else:
            error_details = f"STDOUT:\n{_filter_dbt_output(process.stdout)}\n\nSTDERR:\n{_filter_dbt_output(process.stderr)}"
            return {"status": "Failed", "message": "生成数据预览失败。", "error_details": error_details}
    except Exception as e:
        return {"status": "Failed", "message": f"执行数据预览时发生未知错误: {e}"}

@tool
def get_table_and_field_info(prompt_info: str) -> dict:
    """获取指定模型的表和字段信息。"""
    logger.info(f"\n--- 正在调用工具 [get_table_and_field_info] for prompt_info： '{prompt_info}' ---")
    try:
        return {"status": "Success", "table_info": get_table_and_field_info2(prompt_info)}
    except Exception as e:
        return {"status": "Failed", "message": str(e)}

class ScheduleReportUpdateArgs(BaseModel):
    task_name: str = Field(description="定时任务的显示名称。")
    report_name: str = Field(description="需要定时运行的报告的名称。")
    cron_expression: str = Field(description="任务执行时间的Cron表达式。")

# [修改] 工具名和描述
@tool(args_schema=ScheduleReportUpdateArgs)
def schedule_report_update(task_name: str, report_name: str, cron_expression: str) -> dict:
    """调用外部API来为报告创建一个定时调度任务。"""
    logger.info(f"\n--- 正在调用工具 [schedule_report_update] for report '{report_name}' ---")
    # [不变] 内部API调用逻辑不变
    try:
        login_resp = requests.post("http://192.168.47.24:30027/admin-api/system/auth/login", json={"tenantName": "芋道源码", "username": "bossai", "password": "123456"}, headers={"Content-Type": "application/json"}, timeout=10)
        access_token = login_resp.json().get("data", {}).get("accessToken")
        payload = {"name": task_name, "handlerName": "procedureSyncJob", "handlerParam": report_name, "cronExpression": cron_expression, "retryCount": "0", "retryInterval": "0"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
        response = requests.post(ReportAgentConfig.SCHEDULER_API_ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        resp_data = response.json()
        if resp_data.get("code") != 0:
            return {"status": "Failed", "message": f"调度API错误: {resp_data.get('msg', '未知错误')}"}
        return {"status": "Success", "message": "定时任务已成功创建。"}
    except requests.exceptions.RequestException as e:
        return {"status": "Failed", "message": f"调用API失败: {e}"}

class ProposeUniqueReportNameArgs(BaseModel):
    base_name: str = Field(description="建议的报告基础名称，例如 'fct_orders'。")

def get_existing_report_names() -> List[str]:
    """[内部函数] 从manifest中获取已存在的报告列表。"""
    manifest = get_manifest_content()
    return sorted(list(set(node_info.get('name') for node_info in manifest.get('nodes', {}).values() if node_info.get('resource_type') == 'model')))

# [修改] 工具名和描述
@tool(args_schema=ProposeUniqueReportNameArgs)
def propose_unique_report_name(base_name: str) -> dict:
    """
    接收一个建议的报告基础名称，检查其是否存在。如果存在，则自动添加版本后缀 (_v2, _v3, ...) 直到找到一个唯一的名称为止。
    """
    logger.info(f"\n--- 正在调用工具 [propose_unique_report_name] for base_name '{base_name}' ---")
    try:
        existing_names_set = set(get_existing_report_names())
        if base_name not in existing_names_set: return {"status": "Success", "unique_name": base_name}
        version = 2
        while True:
            new_name = f"{base_name}_v{version}"
            if new_name not in existing_names_set: return {"status": "Success", "unique_name": new_name}
            version += 1
    except Exception as e:
        return {"status": "Failed", "message": str(e)}

# [修改] 工具列表
tools = [
    get_table_and_field_info,
    create_report_files,
    validate_report_logic,
    preview_report_data,
    schedule_report_update,
    propose_unique_report_name,
    get_upstream_column_descriptions
]

# ==============================================================================
# 4. LLM 初始化
# ==============================================================================
llm = ChatDashscope(model=ReportAgentConfig.LLM_MODEL_NAME, api_key=ReportAgentConfig.LLM_API_KEY,
                    base_url=ReportAgentConfig.LLM_BASE_URL)

# ==============================================================================
# 5. Agent 构建
# ==============================================================================
def get_simplified_system_context(manifest_path: str) -> str:
    """从manifest.json中提取简化的、对LLM友好的上下文。"""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f: manifest = json.load(f)
        models = sorted(list(set(v['name'] for k, v in manifest.get("nodes", {}).items() if v['resource_type'] == 'model')))
        sources = sorted(list(set((v['source_name'], v['name']) for k, v in manifest.get("sources", {}).items())))
        context = {
            "existing_analysis_modules": models,
            "sources": [{"source_name": s[0], "table_name": s[1]} for s in sources]
        }
        return json.dumps(context, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"

system_context = get_simplified_system_context(ReportAgentConfig.DBT_MANIFEST_PATH)
logger.info(f"提取的系统上下文: {system_context}")

final_system_prompt = ASSEMBLED_PROMPT_TEMPLATE.format(system_context=system_context)

# [修改] Agent图变量名
analysis_agent_graph = create_react_agent(llm, tools, prompt=final_system_prompt)
logger.info("✅ 报告分析 Agent 图已编译并赋值给 'report_agent_graph' 变量。")

# ==============================================================================
# 6. 主函数 (用于本地独立测试)
# ==============================================================================
async def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    logger.info("--- ReAct Agent 本地测试启动 (输入 'exit' 退出) ---")
    while True:
        try:
            user_input = input(f"\n你> ")
            if user_input.lower() in ["exit", "退出"]: break
            inputs = {"messages": [HumanMessage(content=user_input)]}
            async for event in analysis_agent_graph.astream(inputs, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    print(f"\n助手> {last_message.content}")
        except (KeyboardInterrupt, EOFError): break
    logger.info("\n再见！")

if __name__ == "__main__":
    if not os.path.exists(ReportAgentConfig.DBT_MANIFEST_PATH):
        logger.warning(f"警告: {ReportAgentConfig.DBT_MANIFEST_PATH} 未找到。请在dbt项目目录下运行 'dbt compile'。")
    asyncio.run(main())
