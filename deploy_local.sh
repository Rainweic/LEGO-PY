#!/bin/bash

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 没有颜色

# 检查 Python 环境是否已激活
# if [ -z "$VIRTUAL_ENV" ]; then
#     echo -e "${YELLOW}警告: Python 虚拟环境未激活。${NC}"
#     echo -e "${YELLOW}请激活您的虚拟环境以确保依赖项正确。${NC}"
# fi

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)

# # 定义日志文件路径
# LOG_FILE="server.log"

# # 启动服务器并将输出重定向到日志文件
# echo -e "${GREEN}启动服务器并将日志写入 $LOG_FILE${NC}"
# python web/server.py >> $LOG_FILE 2>&1 &

echo -e "${GREEN}启动服务器"
python web/server.py

# 获取服务器的进程 ID
SERVER_PID=$!

# 输出服务器启动信息
echo -e "${GREEN}服务器已启动，进程 ID: $SERVER_PID${NC}"
echo -e "${GREEN}查看日志文件以获取详细信息: $LOG_FILE${NC}"

# 捕获退出信号以便在脚本终止时停止服务器
trap "echo -e '${RED}停止服务器...${NC}'; kill $SERVER_PID; exit 0" SIGINT SIGTERM

# 等待服务器进程
wait $SERVER_PID
