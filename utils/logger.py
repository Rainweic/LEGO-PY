import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, prefix, job_id=None):
    logger = logging.getLogger(f"{prefix}_{name}")
    logger.setLevel(logging.INFO)

    # 使用固定的日志目录
    base_log_dir = '/tmp/lego/log'  # 或者其他您指定的固定路径
    if job_id:
        log_dir = os.path.join(base_log_dir, job_id)
    else:
        log_dir = base_log_dir

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')

    # 文件处理器
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(f'[{prefix}]: %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
