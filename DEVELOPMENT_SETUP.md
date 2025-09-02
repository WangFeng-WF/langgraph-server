# 开发环境设置

## 问题解决

您遇到的错误是由于无法连接到PostgreSQL数据库服务器导致的。我们已经修改了代码以支持本地开发模式。

## 快速启动

### 方法1：使用开发脚本（推荐）

```bash
python start_dev.py
```

### 方法2：使用环境变量

```bash
# Windows PowerShell
$env:USE_LOCAL_STORAGE="true"
$env:DEBUG="true"
langgraph dev

# Windows CMD
set USE_LOCAL_STORAGE=true
set DEBUG=true
langgraph dev

# Linux/Mac
export USE_LOCAL_STORAGE=true
export DEBUG=true
langgraph dev
```

### 方法3：直接运行服务器

```bash
python run_server.py
```

## 配置说明

### 存储模式

- **内存存储** (`USE_LOCAL_STORAGE=true`): 适合开发环境，数据存储在内存中
- **PostgreSQL存储** (`USE_LOCAL_STORAGE=false`): 适合生产环境，需要配置数据库连接

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `USE_LOCAL_STORAGE` | `true` | 是否使用内存存储 |
| `DEBUG` | `false` | 是否启用调试模式 |
| `DASHSCOPE_API_KEY` | `sk-7c61b5435ea94666b3a50d4a0d889bd2` | 通义千问API密钥 |
| `QWEN_MODEL` | `qwen-plus` | 使用的模型名称 |
| `HOST` | `0.0.0.0` | 服务器主机地址 |
| `PORT` | `8123` | 服务器端口 |

## 故障排除

### 数据库连接问题

如果仍然遇到数据库连接问题，请确保：

1. 使用内存存储模式（设置 `USE_LOCAL_STORAGE=true`）
2. 检查网络连接
3. 确认数据库服务器是否运行

### 其他问题

如果遇到其他问题，请检查：

1. Python依赖是否正确安装
2. 环境变量是否正确设置
3. 端口8123是否被占用

## 生产环境部署

当需要部署到生产环境时：

1. 设置 `USE_LOCAL_STORAGE=false`
2. 配置正确的 `DB_URI` 环境变量
3. 确保数据库服务器可访问
4. 设置适当的 `DEBUG=false`
