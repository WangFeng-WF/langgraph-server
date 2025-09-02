# 基于 thread_id 的上下文对话功能使用说明

## 功能概述

本功能为 LangGraph 聊天代理添加了基于 `thread_id` 的上下文对话支持，允许用户在不同的对话线程中保持独立的对话历史，实现真正的上下文感知对话。

## 主要特性

1. **线程隔离**: 每个 `thread_id` 维护独立的对话历史
2. **消息持久化**: 消息自动保存到内存存储中
3. **上下文加载**: 根据 `thread_id` 自动加载历史消息
4. **线程管理**: 支持清除线程历史、获取统计信息等
5. **元数据跟踪**: 记录消息数量、最后更新时间等信息

## 核心组件

### ThreadMessageStore 类

消息存储管理器，负责：
- 保存消息到指定线程
- 根据线程ID检索历史消息
- 清除线程消息
- 提供线程统计信息

### State 类增强

在原有 State 类基础上添加了：
- `thread_id` 字段
- `_load_thread_messages()` 方法：加载线程历史消息
- `clear_thread_history()` 方法：清除线程历史
- `get_thread_stats()` 方法：获取线程统计信息
- 增强的 `add_message()` 方法：自动保存消息到存储

## 使用方法

### 1. 基本对话流程

```python
from src.agent.graph import graph, State
from langchain_core.messages import HumanMessage

# 创建状态，指定 thread_id
state = State(
    user_id="user_001",
    thread_id="conversation_001",
    system_prompt="你是一个有用的AI助手"
)

# 添加用户消息
user_msg = HumanMessage(content="你好，请介绍一下你自己")
state.add_message(user_msg)

# 调用图处理
config = {"configurable": {"thread_id": "conversation_001"}}
result = await graph.ainvoke(
    {"messages": state.messages},
    config=config
)
```

### 2. 继续对话（保持上下文）

```python
# 继续添加消息（会自动加载历史上下文）
user_msg2 = HumanMessage(content="我刚才问了你什么？")
state.add_message(user_msg2)

# 再次调用图处理
result2 = await graph.ainvoke(
    {"messages": state.messages},
    config=config
)
```

### 3. 创建新的对话线程

```python
# 创建新的线程，完全独立的对话
state2 = State(
    user_id="user_002",
    thread_id="conversation_002",
    system_prompt="你是一个数学老师"
)

user_msg3 = HumanMessage(content="请解释微积分")
state2.add_message(user_msg3)

config2 = {"configurable": {"thread_id": "conversation_002"}}
result3 = await graph.ainvoke(
    {"messages": state2.messages},
    config=config2
)
```

### 4. 线程管理操作

```python
# 获取线程统计信息
stats = state.get_thread_stats()
print(f"线程统计: {stats}")

# 清除线程历史
state.clear_thread_history()

# 重新设置线程ID（会加载该线程的历史消息）
state.set_thread_id("existing_thread_id")
```

### 5. 全局存储管理

```python
from src.agent.graph import message_store

# 获取所有线程信息
thread_info = message_store.get_thread_info()
print(f"所有线程: {thread_info}")

# 获取线程总数
thread_count = message_store.get_thread_count()
print(f"总线程数: {thread_count}")

# 清除特定线程
message_store.clear_thread("thread_id_to_clear")
```

## 配置说明

### 环境变量

确保在 `.env` 文件中设置：

```env
DASHSCOPE_API_KEY=your_api_key_here
QWEN_MODEL=qwen-plus
```

### 运行时配置

通过 `config` 参数传递 `thread_id`：

```python
config = {
    "configurable": {
        "thread_id": "your_thread_id"
    }
}
```

## 测试

运行测试文件验证功能：

```bash
python test_thread_context.py
```

测试包括：
- 基本对话功能
- 上下文保持
- 线程隔离
- 线程管理
- 消息持久化

## 注意事项

1. **内存存储**: 当前使用内存存储，重启服务后数据会丢失
2. **线程ID唯一性**: 确保每个对话使用唯一的 `thread_id`
3. **消息格式**: 支持 LangChain 消息格式和字典格式
4. **错误处理**: 包含完整的错误处理和日志记录

## 扩展功能

### 持久化存储

可以扩展 `ThreadMessageStore` 类支持数据库存储：

```python
class DatabaseThreadMessageStore(ThreadMessageStore):
    def __init__(self, db_connection):
        self.db = db_connection
        # 实现数据库操作
```

### 消息过滤

可以添加消息过滤功能，如按时间范围、消息类型等过滤历史消息。

### 线程合并

可以实现线程合并功能，将多个线程的对话历史合并。

## 示例应用场景

1. **多用户聊天**: 每个用户使用独立的 `thread_id`
2. **多主题对话**: 不同主题使用不同的 `thread_id`
3. **会话恢复**: 通过 `thread_id` 恢复之前的对话
4. **A/B测试**: 使用不同的 `thread_id` 测试不同的对话策略
