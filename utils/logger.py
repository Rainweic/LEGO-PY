import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, prefix, job_id=None):
    logger = logging.getLogger(f"{prefix}_{name}")
    
    # 根据环境变量设置日志级别
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level))

    # 使用固定的日志目录
    if job_id:
        log_dir = os.path.join(os.path.dirname(__file__), "../cache", job_id, "logs")
    else:
        log_dir = os.path.join(os.path.dirname(__file__), "../cache/logs")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')

    # 文件处理器 - 使用FileHandler而不是RotatingFileHandler来覆盖原始内容
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    # 格式化器
    formatter = logging.Formatter(f'[%(levelname)s] [{prefix}]: %(asctime)s [%(name)s] %(filename)s:%(lineno)d : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
