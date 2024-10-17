import logging
import importlib
import os
import inspect

# 获取当前目录下的所有模块
stages_dir = os.path.dirname(__file__)
modules = [f[:-3] for f in os.listdir(stages_dir) if f.endswith('.py') and f != '__init__.py']

# 动态导入模块
for module in modules:
    imported_module = importlib.import_module(f".{module}", package=__name__)
    
    # 获取模块中的所有类名
    for name, obj in inspect.getmembers(imported_module, inspect.isclass):
        # 确保类是当前模块的类
        if obj.__module__ == imported_module.__name__:
            globals()[name] = obj

# 自动生成__all__列表，包含所有类名
__all__ = [name for name in globals() if not name.startswith('_')]


def create_stage(stage_type: str, name: str, args: dict, job_id: str = None):
    """
    根据阶段类型、名称和参数创建相应的阶段实例。

    参数:
        stage_type (str): 阶段类型
        name (str): 阶段名称
        args (dict): 阶段参数

    返回:
        实例化的阶段对象
    """
    # 动态获取阶段类
    stage_class = globals().get(stage_type)
    if stage_class is None:
        raise ValueError(f"未知的阶段类型: {stage_type}")

    # 创建阶段实例并返回
    stage_obj = stage_class(**args)

    if name:
        stage_obj.name = name

    # 为这个 stage 实例设置日志记录器
    logger = setup_logger(f"{stage_type}_{name}", job_id=job_id)
    stage_obj.logger = logger

    return stage_obj


def setup_logger(name, job_id=None):
    # 创建日志目录（如果不存在）
    log_dir = os.path.join(os.path.dirname(__file__), '../cache/logs')
    if job_id:
        log_dir = os.path.join(log_dir, job_id)
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
    formatter = logging.Formatter("[STAGES]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # 将格式化器添加到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 在每个stage文件中使用这个函数来设置日志记录器
# 例如，在 my_stage.py 中：
# logger = setup_logger(__name__)
