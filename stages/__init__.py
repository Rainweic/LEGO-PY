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


logging.basicConfig(
    format="[STAGES]: %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

def create_stage(stage_type: str, name: str, args: dict):
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

    return stage_obj