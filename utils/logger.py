import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, prefix, job_id=None):
    logger = logging.getLogger(f"{prefix}_{name}")
    logger.setLevel(logging.INFO)

    # 使用固定的日志目录
    if job_id:
        log_dir = os.path.join(os.path.dirname(__file__), "../cache", job_id, "logs")
    else:
        log_dir = os.path.join(os.path.dirname(__file__), "../cache/logs")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')

    # 文件处理器
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(f'[{prefix}]: %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
