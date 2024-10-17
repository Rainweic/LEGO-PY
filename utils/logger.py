import os
import logging


def setup_logger(name, comp_name="DAG", job_id=None):
    # 创建日志目录（如果不存在）
    
    if job_id:
        log_dir = os.path.join(os.path.dirname(__file__), f'../cache/{job_id}/logs')
    else:
        log_dir = os.path.join(os.path.dirname(__file__), '../cache/logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件路径
    log_file = os.path.join(log_dir, f"{name}.log")

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(f"[{comp_name}]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # 将格式化器添加到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

