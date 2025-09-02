.PHONY: install dev test clean run help

# 默认目标
help:
	@echo "可用的命令:"
	@echo "  install   - 安装依赖"
	@echo "  dev       - 安装开发依赖"
	@echo "  run       - 启动服务器"
	@echo "  test      - 运行测试"
	@echo "  test-api  - 运行API测试"
	@echo "  clean     - 清理缓存文件"
	@echo "  format    - 格式化代码"
	@echo "  lint      - 代码检查"

# 安装依赖
install:
	pip install -e .

# 安装开发依赖
dev:
	pip install -e ".[dev]"

# 启动服务器
run:
	python run_server.py

# 运行测试
test:
	python test_chat.py

# 运行API测试
test-api:
	python test_api.py

# 格式化代码
format:
	ruff format src/

# 代码检查
lint:
	ruff check src/

# 清理缓存文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache

