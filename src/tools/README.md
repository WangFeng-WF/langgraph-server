# Tools 工具包

这个目录包含了LangGraph代理可用的各种工具。

## 可用工具

### safe_sql_execute

安全的MySQL SQL执行工具，提供以下功能：

#### 功能特性

- ✅ **SQL注入防护**: 自动检测和阻止SQL注入攻击
- ✅ **危险操作检测**: 阻止DROP、DELETE、ALTER等危险操作
- ✅ **系统表保护**: 防止对系统表的操作
- ✅ **语法验证**: 基本的SQL语法检查
- ✅ **连接管理**: 自动管理数据库连接
- ✅ **错误处理**: 完善的错误处理和日志记录
- ✅ **执行统计**: 提供执行时间和影响行数统计

#### 支持的操作

- ✅ SELECT 查询
- ✅ SHOW 语句
- ✅ DESCRIBE 语句
- ✅ EXPLAIN 语句

#### 阻止的操作

- ❌ DROP 操作
- ❌ DELETE 操作
- ❌ TRUNCATE 操作
- ❌ ALTER 操作
- ❌ CREATE 操作
- ❌ INSERT 操作
- ❌ UPDATE 操作
- ❌ GRANT 操作
- ❌ REVOKE 操作
- ❌ 系统表操作

#### 使用方法

##### 1. 直接调用函数

```python
from tools.safe_sql_execute import safe_sql_execute

# 执行安全查询
result = safe_sql_execute(
    sql="SELECT * FROM users WHERE id = 1",
    host="localhost",
    port=3306,
    user="root",
    password="password",
    database="test_db"
)

print(result)
```

##### 2. 使用LangChain工具

```python
from tools.safe_sql_execute import safe_sql_execute_tool

# 准备输入参数
tool_input = {
    "sql": "SELECT * FROM users WHERE id = 1",
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password",
    "database": "test_db"
}

# 执行工具
result = safe_sql_execute_tool._run(json.dumps(tool_input))
print(result)
```

##### 3. 在代理中使用

```python
from agent.graph import create_agent_with_tools

# 创建带有工具的代理
agent_executor = create_agent_with_tools()

# 使用代理执行查询
result = await agent_executor.ainvoke({
    "input": "请查询用户表中id为1的用户信息",
    "chat_history": []
})
```

#### 返回格式

```json
{
    "success": true,
    "message": "查询成功，返回 1 条记录",
    "sql": "SELECT * FROM users WHERE id = 1",
    "sql_type": "SELECT",
    "execution_time": 0.123,
    "security_check": "PASSED",
    "data": [
        {
            "id": 1,
            "name": "张三",
            "email": "zhangsan@example.com"
        }
    ],
    "row_count": 1
}
```

#### 错误处理

当检测到危险操作时：

```json
{
    "success": false,
    "message": "检测到危险操作，已阻止执行",
    "sql": "DROP TABLE users",
    "security_check": "FAILED"
}
```

当SQL语法错误时：

```json
{
    "success": false,
    "message": "SQL语法验证失败",
    "sql": "INVALID SQL",
    "security_check": "FAILED"
}
```

#### 配置要求

确保已安装必要的依赖：

```bash
pip install pymysql>=1.1.0
```

#### 安全建议

1. **最小权限原则**: 使用具有最小必要权限的数据库用户
2. **网络隔离**: 确保数据库服务器在网络层面得到适当保护
3. **定期审计**: 定期检查工具的使用日志
4. **参数验证**: 在应用层面验证所有输入参数
5. **连接池**: 在生产环境中考虑使用连接池

#### 测试

运行测试脚本：

```bash
python test_safe_sql_tool.py
```

## 添加新工具

要添加新工具，请按照以下步骤：

1. 在 `src/tools/` 目录下创建新的工具文件
2. 实现工具功能，继承 `BaseTool` 类
3. 在 `src/tools/__init__.py` 中导入新工具
4. 在 `src/agent/graph.py` 中集成新工具
5. 更新系统提示词，说明新工具的用途
6. 编写测试用例

## 注意事项

- 所有工具都应该包含适当的安全检查
- 工具应该提供清晰的错误信息
- 工具应该支持异步操作
- 工具应该包含适当的日志记录
- 工具应该提供详细的文档说明
