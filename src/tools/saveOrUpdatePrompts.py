# -*- coding: utf-8 -*-
"""
保存或更新提示词到私有库

工作流程：接收参数→大模型结构化输出→查询是否存在→决定INSERT/UPDATE→执行SQL→返回结果
工具：MCP工具，需调用MCP服务器上的selectBySql和执行SQL工具
大模型：通义千问的qwen-max-latest
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MCP服务器配置
MCP_SERVER_CONFIG = {
    "server_type": "sse",
    "url": "http://192.168.58.77:8088/sse",
    "timeout_seconds": 60
}


class AiPromptSave(BaseModel):
    """AI提示词保存参数模型"""
    operation_type: Optional[str] = Field(None, description="操作类型：新增或修改")
    title: Optional[str] = Field(None, description="指标/提示词标题")
    type: Optional[str] = Field(None, description="指标类型")
    fields: Optional[str] = Field(None, description="业务域")
    instruction: Optional[str] = Field(None, description="用法说明")
    inputs: Optional[str] = Field(None, description="输入参数示例")
    sql_example: Optional[str] = Field(None, description="SQL示例")
    creator: Optional[str] = Field(default="system", description="创建者")
    organization: Optional[str] = Field(None, description="组织")
    user: Optional[str] = Field(None, description="用户")

    def build_check_conditions(self) -> str:
        """构建检查是否存在的WHERE条件"""
        conditions = []

        if self.title and self.title.strip():
            conditions.append(f"title = '{self.title}'")
        if self.type and self.type.strip():
            conditions.append(f"type = '{self.type}'")
        if self.fields and self.fields.strip():
            conditions.append(f"fields = '{self.fields}'")

        return " AND ".join(conditions) if conditions else ""

    def build_insert_sql(self) -> str:
        """构建INSERT语句"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        fields = []
        values = []

        if self.title:
            fields.append("title")
            values.append(f"'{self.title}'")
        if self.instruction:
            fields.append("instruction")
            values.append(f"'{self.instruction}'")
        if self.type:
            fields.append("type")
            values.append(f"'{self.type}'")
        if self.fields:
            fields.append("fields")
            values.append(f"'{self.fields}'")
        if self.inputs:
            fields.append("inputs")
            values.append(f"'{self.inputs}'")
        if self.sql_example:
            fields.append("sql_example")
            values.append(f"'{self.sql_example}'")
        if self.creator:
            fields.append("creator")
            values.append(f"'{self.creator}'")
        if self.organization:
            fields.append("organization")
            values.append(f"'{self.organization}'")
        if self.user:
            fields.append("user")
            values.append(f"'{self.user}'")

        # 添加时间字段
        fields.extend(["create_time", "update_time", "deleted"])
        values.extend([f"'{current_time}'", f"'{current_time}'", "0"])

        return f"INSERT INTO ai_prompt ({', '.join(fields)}) VALUES ({', '.join(values)})"

    def build_update_sql(self, existing_id: int) -> str:
        """构建UPDATE语句（title不能更新）"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        updates = []

        if self.instruction:
            updates.append(f"instruction = '{self.instruction}'")
        if self.type:
            updates.append(f"type = '{self.type}'")
        if self.fields:
            updates.append(f"fields = '{self.fields}'")
        if self.inputs:
            updates.append(f"inputs = '{self.inputs}'")
        if self.sql_example:
            updates.append(f"sql_example = '{self.sql_example}'")
        if self.creator:
            updates.append(f"updater = '{self.creator}'")
        if self.organization:
            updates.append(f"organization = '{self.organization}'")
        if self.user:
            updates.append(f"user = '{self.user}'")

        # 添加更新时间
        updates.append(f"update_time = '{current_time}'")

        return f"UPDATE ai_prompt SET {', '.join(updates)} WHERE id = {existing_id}"

import os
from src.tools.dashscope import ChatDashscope

class AppConfig:
    ALLOWED_BUSINESS_DOMAINS = ["生产", "采购", "销售", "供应链"]
    LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ca949c46e4904479927923a41562d4d3")
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME = "qwen-max-latest"

    STEP_MAP = {
        1: "名词定义", 2: "计算公式", 3: "分析维度与指标", 4: "数据来源",
        5: "指标计算-结果查询SQL", 6: "明细计算-实时计算SQL",
        7: "指标分析方法", 8: "明细分析方法"
    }

try:
    merged_conf = {'api_key': AppConfig.LLM_API_KEY,
                   'base_url': AppConfig.LLM_BASE_URL,
                   'extra_body': {'enable_thinking': False}, 'max_retries': 3, 'model': AppConfig.LLM_MODEL_NAME}
    llm = ChatDashscope(**merged_conf)
    logger.info("通义千问模型 (qwen-max) 初始化成功。")
except Exception as e:
    logger.error(f"模型初始化失败，请检查API KEY。错误: {e}")
    exit()

async def parse_prompts_to_structured_output(prompts_info: str) -> AiPromptSave:
    """使用大模型的结构化输出解析提示词信息"""
    try:
        parse_prompt = f"""
请解析以下用户输入的提示词信息，提取出所有相关字段：

用户输入：
{prompts_info}

注意：
1. 操作类型从输入中识别（新增、修改、添加、更新等）
2. 提取的字段值不要包含【】符号
3. 严格按照用户输入提取，不要添加额外内容
4. SQL示例中的注释需要保留
5. 如果某个字段没有提到，设置为null
        """

        structured_llm = llm.with_structured_output(AiPromptSave)
        result = structured_llm.invoke([HumanMessage(content=parse_prompt)])

        logger.info(f"Parsed structured output: {result}")
        return result

    except Exception as e:
        logger.error(f"Error parsing to structured output: {e}")
        return manual_parse_prompts(prompts_info)


def manual_parse_prompts(prompts_info: str) -> AiPromptSave:
    """手动解析提示词信息的降级方法"""
    operation_type = None
    title = None
    type_val = None
    fields = None
    instruction = None
    inputs = None
    sql_example = None
    organization = None
    user = None

    # 检测操作类型
    if any(keyword in prompts_info for keyword in ["新增", "添加", "新建"]):
        operation_type = "新增"
    elif any(keyword in prompts_info for keyword in ["修改", "更新", "编辑"]):
        operation_type = "修改"

    lines = prompts_info.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '【指标/提示词标题】' in line or '【标题】' in line:
            title = line.split('】')[1].strip() if '】' in line else None
        elif '【指标类型】' in line or '【类型】' in line:
            type_val = line.split('】')[1].strip() if '】' in line else None
        elif '【业务域】' in line:
            fields = line.split('】')[1].strip() if '】' in line else None
        elif '【用法说明】' in line or '【说明】' in line:
            instruction = line.split('】')[1].strip() if '】' in line else None
        elif '【输入参数示例】' in line or '【输入参数】' in line:
            inputs = line.split('】')[1].strip() if '】' in line else None
        elif '【SQL示例】' in line or '【SQL 示例】' in line:
            sql_example = line.split('】')[1].strip() if '】' in line else None
        elif '【组织】' in line:
            organization = line.split('】')[1].strip() if '】' in line else None
        elif '【用户】' in line:
            user = line.split('】')[1].strip() if '】' in line else None

    return AiPromptSave(
        operation_type=operation_type,
        title=title,
        type=type_val,
        fields=fields,
        instruction=instruction,
        inputs=inputs,
        sql_example=sql_example,
        organization=organization,
        user=user
    )


async def check_prompt_exists(ai_prompt: AiPromptSave) -> Dict[str, Any]:
    """检查提示词是否已存在"""
    try:
        where_conditions = ai_prompt.build_check_conditions()
        if not where_conditions:
            return {"exists": False, "error": "无法构建查询条件"}

        check_sql = f"SELECT id, title FROM ai_prompt WHERE {where_conditions} AND deleted = 0"

        async with sse_client(url=MCP_SERVER_CONFIG["url"]) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                result = await session.call_tool("selectBySql", arguments={"sql": check_sql})

                if result.content and result.content[0].text:
                    # 解析查询结果 [TextContent(type='text', text='"查询结果为空"', annotations=None, meta=None)]
                    result_text = result.content[0].text
                    if "[]" in result_text or "empty" in result_text.lower() or "查询结果为空" in result_text:
                        return {"exists": False, "data": None}
                    else:
                        # 尝试解析出ID
                        # 数据格式为：'"[{\\"id\\":494,\\"title\\":\\"生产计划达成率（测试）\\"}]"'
                        try:
                            import re
                            match = re.search(r'\[.*\]', result_text)
                            if match:
                                json_str = match.group(0).replace('\\"', '"')
                                records = json.loads(json_str)
                                if records and isinstance(records, list) and "id" in records[0]:
                                    return {"exists": True, "data": result_text, "id": records[0]["id"]}
                            return {"exists": True, "data": result_text, "id": None}
                        except:
                            return {"exists": True, "data": result_text, "id": None}
                else:
                    return {"exists": False, "data": None}

    except Exception as e:
        logger.error(f"Error checking prompt existence: {e}")
        return {"exists": False, "error": str(e)}


async def execute_sql_with_mcp(sql: str) -> Dict[str, Any]:
    """使用MCP工具执行SQL语句"""
    try:
        async with sse_client(url=MCP_SERVER_CONFIG["url"]) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()

                # 根据SQL类型选择合适的工具 selectBySql、insertBySql、updateBySql，暂时只支持这三种
                if sql.upper().startswith("SELECT"):
                    result = await session.call_tool("selectBySql", arguments={"sql": sql})
                elif sql.upper().startswith("INSERT"):
                    result = await session.call_tool("insertBySql", arguments={"sql": sql})
                elif sql.upper().startswith("UPDATE"):
                    result = await session.call_tool("updateBySql", arguments={"sql": sql})
                else:
                    return {
                        "success": False,
                        "error": "仅支持SELECT、INSERT、UPDATE语句"
                    }

                return {
                    "success": True,
                    "data": result.content[0].text if result.content else None,
                    "sql_executed": sql,
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Error executing SQL with MCP: {e}")
        return {
            "success": False,
            "error": f"MCP工具调用失败: {str(e)}",
            "error_type": type(e).__name__
        }

async def save_or_update_prompts(ai_prompt: AiPromptSave) -> Dict[str, Any]:
    """
    使用结构化输出的保存或更新函数

    Args:
        prompts_info: 用户输入的提示词信息

    Returns:
        Dict包含操作结果或错误信息
    """
    try:
        # logger.info(f"Starting structured save/update for prompts: {prompts_info}")

        # # 步骤1: 使用大模型解析参数到结构化输出
        # ai_prompt = await parse_prompts_to_structured_output(prompts_info)

        # 步骤2: 检查是否存在相同的提示词
        check_result = await check_prompt_exists(ai_prompt)

        if "error" in check_result:
            return {
                "success": False,
                "message": "检查提示词存在性失败",
                "parsed_params": ai_prompt.dict(),
                "error": check_result["error"],
                "timestamp": datetime.now().isoformat()
            }

        # 步骤3: 根据检查结果决定操作类型
        if check_result["exists"]:
            # 存在则更新
            existing_id = check_result.get("id")
            if existing_id is None:
                return {
                    "success": False,
                    "message": "无法获取现有记录的ID",
                    "parsed_params": ai_prompt.dict(),
                    "timestamp": datetime.now().isoformat()
                }

            sql = ai_prompt.build_update_sql(existing_id)
            operation = "更新"
        else:
            # 不存在则插入
            sql = ai_prompt.build_insert_sql()
            operation = "新增"

        # 步骤4: 执行SQL
        execute_result = await execute_sql_with_mcp(sql)

        if execute_result["success"]:
            return {
                "success": True,
                "message": f"提示词{operation}成功",
                "operation": operation,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "execute_result": execute_result["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"提示词{operation}失败",
                "operation": operation,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "error": execute_result["error"],
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error in structured save/update: {e}")
        return {
            "success": False,
            "message": "结构化保存/更新过程出错",
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }

async def save_or_update_prompts_structured(prompts_info: str) -> Dict[str, Any]:
    """
    使用结构化输出的保存或更新函数

    Args:
        prompts_info: 用户输入的提示词信息

    Returns:
        Dict包含操作结果或错误信息
    """
    try:
        logger.info(f"Starting structured save/update for prompts: {prompts_info}")

        # 步骤1: 使用大模型解析参数到结构化输出
        ai_prompt = await parse_prompts_to_structured_output(prompts_info)

        # 步骤2: 检查是否存在相同的提示词
        check_result = await check_prompt_exists(ai_prompt)

        if "error" in check_result:
            return {
                "success": False,
                "message": "检查提示词存在性失败",
                "prompts_info": prompts_info,
                "parsed_params": ai_prompt.dict(),
                "error": check_result["error"],
                "timestamp": datetime.now().isoformat()
            }

        # 步骤3: 根据检查结果决定操作类型
        if check_result["exists"]:
            # 存在则更新
            existing_id = check_result.get("id")
            if existing_id is None:
                return {
                    "success": False,
                    "message": "无法获取现有记录的ID",
                    "prompts_info": prompts_info,
                    "parsed_params": ai_prompt.dict(),
                    "timestamp": datetime.now().isoformat()
                }

            sql = ai_prompt.build_update_sql(existing_id)
            operation = "更新"
        else:
            # 不存在则插入
            sql = ai_prompt.build_insert_sql()
            operation = "新增"

        # 步骤4: 执行SQL
        execute_result = await execute_sql_with_mcp(sql)

        if execute_result["success"]:
            return {
                "success": True,
                "message": f"提示词{operation}成功",
                "operation": operation,
                "prompts_info": prompts_info,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "execute_result": execute_result["data"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"提示词{operation}失败",
                "operation": operation,
                "prompts_info": prompts_info,
                "parsed_params": ai_prompt.dict(),
                "generated_sql": sql,
                "error": execute_result["error"],
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error in structured save/update: {e}")
        return {
            "success": False,
            "message": "结构化保存/更新过程出错",
            "prompts_info": prompts_info,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def format_save_result(result: Dict[str, Any]) -> str:
    """格式化保存结果为可读的字符串"""
    if not result["success"]:
        return f"操作失败: {result.get('error', '未知错误')}"

    operation = result.get("operation", "操作")
    return f"{operation}成功: {result.get('execute_result', '操作完成')}"


# 同步包装器函数，供其他模块调用
def save_or_update_prompts_sync(prompts_info: str) -> Dict[str, Any]:
    """同步版本的保存更新函数"""
    try:
        return asyncio.run(save_or_update_prompts_structured(prompts_info))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        return {
            "success": False,
            "message": "同步调用失败",
            "prompts_info": prompts_info,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def test_save_functionality():
    """测试保存功能"""
    test_cases = [
        """新增 【指标/提示词标题】生产计划达成率（测试）
【指标类型】指标计算-结果查询SQL
【业务域】生产
【用法说明】用于计算和查询生产计划的达成情况
【输入参数示例】startDate, endDate
【SQL示例】SELECT plan_rate FROM production_plan WHERE date BETWEEN ? AND ?""",

        """修改 【指标/提示词标题】生产计划达成率（测试）
【指标类型】指标计算-结果查询SQL
【业务域】生产
【用法说明】更新后的用法说明：用于计算生产计划达成率
【输入参数示例】startDate, endDate, factoryId
【SQL示例】SELECT plan_rate FROM production_plan WHERE date BETWEEN ? AND ? AND factory_id = ?"""
    ]

    print("开始测试提示词保存/更新功能...")
    print("=" * 60)

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:\n{test_input}")
        print("-" * 40)

        try:
            result = save_or_update_prompts_sync(test_input)
            print(f"操作结果: {format_save_result(result)}")
        except Exception as e:
            print(f"测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("测试完成")


if __name__ == "__main__":
    test_save_functionality()
