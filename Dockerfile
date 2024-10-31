FROM python:3.10-slim

WORKDIR /app

ENV LANG C.UTF-8

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt


# 启动命令
CMD ["sh", "deploy_local.sh"]