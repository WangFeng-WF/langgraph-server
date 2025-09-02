"""安全的MySQL SQL执行工具。

该工具提供安全的MySQL数据库操作功能，包括：
1. SQL注入防护
2. 危险操作检测
3. 执行结果验证
4. 错误处理和日志记录
"""

import re
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import pymysql
from pymysql.cursors import DictCursor
from langchain.tools import BaseTool
from langchain.schema import BaseOutputParser
import json
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SQLExecutionResult:
    """SQL执行结果数据类"""
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    affected_rows: int = 0
    execution_time: float = 0.0
    sql_type: str = "UNKNOWN"


class SQLSecurityChecker:
    """SQL安全检查器"""
    
    # 危险操作关键词
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        'GRANT', 'REVOKE', 'EXECUTE', 'EXEC', 'EXECUTE IMMEDIATE'
    ]
    
    # 系统表前缀
    SYSTEM_TABLES = [
        'mysql.', 'information_schema.', 'performance_schema.', 'sys.'
    ]
    
    @classmethod
    def is_dangerous_operation(cls, sql: str) -> bool:
        """检查是否为危险操作"""
        sql_upper = sql.upper().strip()
        
        # 检查危险关键词
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return True
        
        # 检查系统表操作
        for table_prefix in cls.SYSTEM_TABLES:
            if table_prefix in sql_upper:
                return True
        
        return False
    
    @classmethod
    def is_select_only(cls, sql: str) -> bool:
        """检查是否为只读查询"""
        sql_upper = sql.upper().strip()
        return sql_upper.startswith('SELECT')
    
    @classmethod
    def validate_sql_syntax(cls, sql: str) -> bool:
        """简单的SQL语法验证"""
        sql = sql.strip()
        if not sql:
            return False
        
        # 检查基本语法结构
        if not re.match(r'^(SELECT|SHOW|DESCRIBE|EXPLAIN)', sql.upper()):
            return False
        
        return True


class MySQLConnectionManager:
    """MySQL连接管理器"""
    
    def __init__(self, host: str = None, port: int = None, 
                 user: str = None, password: str = None, database: str = None):
        # 从环境变量获取默认配置
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '3306'))
        self.user = user or os.getenv('DB_USER', 'root')
        self.password = password or os.getenv('DB_PASSWORD', '')
        self.database = database or os.getenv('DB_DATABASE', '')
        self.connection = None
    
    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True
            )
            logger.info(f"成功连接到MySQL数据库: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接MySQL数据库失败: {e}")
            return False
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("MySQL数据库连接已关闭")
    
    def execute_query(self, sql: str) -> SQLExecutionResult:
        """执行SQL查询"""
        import time
        start_time = time.time()
        
        try:
            if not self.connection or not self.connection.open:
                if not self.connect():
                    return SQLExecutionResult(
                        success=False,
                        message="无法连接到数据库"
                    )
            
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                
                if sql.strip().upper().startswith('SELECT'):
                    # 查询操作
                    data = cursor.fetchall()
                    execution_time = time.time() - start_time
                    
                    return SQLExecutionResult(
                        success=True,
                        message=f"查询成功，返回 {len(data)} 条记录",
                        data=data,
                        affected_rows=len(data),
                        execution_time=execution_time,
                        sql_type="SELECT"
                    )
                else:
                    # 其他操作
                    affected_rows = cursor.rowcount
                    execution_time = time.time() - start_time
                    
                    return SQLExecutionResult(
                        success=True,
                        message=f"操作成功，影响 {affected_rows} 行",
                        affected_rows=affected_rows,
                        execution_time=execution_time,
                        sql_type="OTHER"
                    )
                    
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SQL执行失败: {e}")
            
            return SQLExecutionResult(
                success=False,
                message=f"SQL执行失败: {str(e)}",
                execution_time=execution_time
            )


def safe_sql_execute(sql: str, 
                    host: str = None, 
                    port: int = None,
                    user: str = None, 
                    password: str = None,
                    database: str = None) -> Dict[str, Any]:
    """
    安全执行MySQL SQL语句
    
    Args:
        sql: 要执行的SQL语句
        host: MySQL主机地址
        port: MySQL端口
        user: MySQL用户名
        password: MySQL密码
        database: 数据库名称
    
    Returns:
        包含执行结果的字典
    """
    
    # 安全检查
    if SQLSecurityChecker.is_dangerous_operation(sql):
        return {
            "success": False,
            "message": "检测到危险操作，已阻止执行",
            "sql": sql,
            "security_check": "FAILED"
        }
    
    if not SQLSecurityChecker.validate_sql_syntax(sql):
        return {
            "success": False,
            "message": "SQL语法验证失败",
            "sql": sql,
            "security_check": "FAILED"
        }
    
    # 创建连接管理器
    conn_manager = MySQLConnectionManager(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    
    try:
        # 执行SQL
        result = conn_manager.execute_query(sql)
        
        # 格式化返回结果
        response = {
            "success": result.success,
            "message": result.message,
            "sql": sql,
            "sql_type": result.sql_type,
            "execution_time": round(result.execution_time, 3),
            "security_check": "PASSED"
        }
        
        if result.success:
            if result.data:
                # 查询结果
                response["data"] = result.data
                response["row_count"] = len(result.data)
            else:
                # 其他操作结果
                response["affected_rows"] = result.affected_rows
        
        return response
        
    finally:
        # 确保连接被关闭
        conn_manager.disconnect()


class SafeSQLExecuteTool(BaseTool):
    """安全的MySQL SQL执行工具"""
    
    name = "safe_sql_execute"
    description = """
    安全执行MySQL SQL查询语句。支持SELECT查询操作，具有SQL注入防护和危险操作检测功能。
    
    输入格式：
    {
        "sql": "SELECT * FROM table_name WHERE condition",
        "host": "localhost",        # 可选，默认从.env文件的DB_HOST读取
        "port": 3306,              # 可选，默认从.env文件的DB_PORT读取
        "user": "root",            # 可选，默认从.env文件的DB_USER读取
        "password": "password",    # 可选，默认从.env文件的DB_PASSWORD读取
        "database": "database_name" # 可选，默认从.env文件的DB_DATABASE读取
    }
    
    环境变量配置（.env文件）：
    DB_HOST=localhost
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=your_password
    DB_DATABASE=your_database
    
    注意：
    1. 只支持SELECT查询操作
    2. 会自动检测和阻止危险操作
    3. 包含SQL注入防护
    4. 如果未提供连接参数，将从.env文件读取默认配置
    """
    
    def _run(self, sql_input: str) -> str:
        """执行工具"""
        try:
            # 解析输入
            if isinstance(sql_input, str):
                # 尝试解析JSON
                try:
                    params = json.loads(sql_input)
                except json.JSONDecodeError:
                    # 如果不是JSON，假设是纯SQL
                    params = {"sql": sql_input}
            else:
                params = sql_input
            
            # 获取参数，如果没有提供则使用环境变量默认值
            sql = params.get("sql", "")
            host = params.get("host")  # 如果为None，会在safe_sql_execute中使用环境变量
            port = params.get("port")  # 如果为None，会在safe_sql_execute中使用环境变量
            user = params.get("user")  # 如果为None，会在safe_sql_execute中使用环境变量
            password = params.get("password")  # 如果为None，会在safe_sql_execute中使用环境变量
            database = params.get("database")  # 如果为None，会在safe_sql_execute中使用环境变量
            
            # 执行SQL
            result = safe_sql_execute(
                sql=sql,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return json.dumps({
                "success": False,
                "message": f"工具执行失败: {str(e)}"
            }, ensure_ascii=False)
    
    async def _arun(self, sql_input: str) -> str:
        """异步执行工具"""
        return self._run(sql_input)


# 创建工具实例
safe_sql_execute_tool = SafeSQLExecuteTool()
