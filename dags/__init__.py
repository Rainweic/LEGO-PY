"""
This initialisation script it called the first time pydags is imported.
Currently, the only purpose of the script is to set up the root logger.
"""

import logging

logging.basicConfig(
    format="[DAG]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 确保 WARNING 和 ERROR 级别的日志也会被输出
logging.getLogger().setLevel(logging.INFO)

# # 可选：如果您想要更细粒度的控制，可以为不同的日志级别设置不同的处理器
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter("[DAG]: %(asctime)s - %(name)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)

# # 获取根日志记录器并添加处理器
# root_logger = logging.getLogger()
# root_logger.addHandler(console_handler)
