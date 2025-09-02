本地或服务器需要python环境，安装langgraph-cli
pip install langgraph-cli

## 开发启动
langgraph dev

# 服务器地址192.168.58.74
# 项目地址 /home/kotl/langGraph/langgraph-server
# 构建镜像 代码修改后需要重新构建镜像。
# 进入/home/kotl/langGraph/langgraph-server
langgraph build -t langgraph-app



# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```
