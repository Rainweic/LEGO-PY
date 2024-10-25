"""
This module contains the base classes of the Stage functionality for pydags,
as well as various implementations of Stages to be used in different contexts.
Most implementations can be subclassed and extended by the user. All Stages
must contain 1) a 'name' property and 2) 'run' method.

There are two primary ways for users to define Stages in their own pipelines.
The first is by decorating a function with the pydags.stage.stage decorator.
The other is by subclassing Stage, SQLiteStage, DiskCacheStage.
"""

from abc import ABC, abstractmethod

import time
import logging
import os
import pickle
import hashlib
import aiofiles

import pyarrow as pa
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from tqdm import tqdm

from .cache import SQLiteCache
from .serialization import PickleSerializer
from .cache import SQLiteCache
from utils.logger import setup_logger


LARGE_DATA_PATH = "./cache"


def serialize(data):
    """根据数据类型选择合适的序列化方法，并记录时间"""
    start_time = time.time()
    if isinstance(data, pd.DataFrame):
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(data), buffer)
        serialized_data = buffer.getvalue()
    elif isinstance(data, pl.DataFrame):
        buffer = pa.BufferOutputStream()
        pq.write_table(data.to_arrow(), buffer)
        serialized_data = buffer.getvalue()
    elif isinstance(data, pl.LazyFrame):
        buffer = pa.BufferOutputStream()
        pq.write_table(data.collect().to_arrow(), buffer)
        serialized_data = buffer.getvalue()
    else:
        serialized_data = pickle.dumps(data)
    elapsed_time = time.time() - start_time
    logging.info(f"Serialization time: {elapsed_time:.2f} seconds")
    return serialized_data


def deserialize(buffer, data_type):
    """根据数据类型选择合适的反序列化方法，并记录时间"""
    start_time = time.time()
    if data_type == 'pandas':
        table = pq.read_table(pa.BufferReader(buffer))
        data = table.to_pandas()
    elif data_type == 'polars':
        table = pq.read_table(pa.BufferReader(buffer))
        data = pl.from_arrow(table)
    else:
        data = pickle.loads(buffer)
    elapsed_time = time.time() - start_time
    logging.info(f"Deserialization time: {elapsed_time:.2f} seconds")
    return data



class Stage(ABC):
    """
    Base abstract class from which all stages must inherit. All subclasses must
    implement at least `run` methods.

    preceding_stages: List of preceding stages for the stage
    name: Name of the stage
    """

    def __init__(self):
        self.preceding_stages = list()
        self._name = None

    def after(self, pipeline_stages: list):
        """Method to add stages as dependencies for the current stage."""
        if not isinstance(pipeline_stages, list):
            pipeline_stages = [pipeline_stages]
        self.preceding_stages.extend(pipeline_stages)
        return self

    @property
    @abstractmethod
    def name(self) -> str: 
        return self._name

    @abstractmethod
    def run(self, *args, **kwargs): ...


class BaseStage(Stage, PickleSerializer, SQLiteCache):
    """
    Stage type to use if a local database cache (i.e. SQLite) is required.
    SQLite can be used to pass data between stages, or cache values to be used
    elsewhere downstream. It's completely up to the implementer/user, as this
    interface to SQLite is generic, and enables the reading/writing of generic
    Python objects from/to SQLite through pickle-based serialization.

    The underlying DAG of the Pipeline object requires serialisation itself as
    part of the inner workings of pydags.
    """

    input_data_names: list = []  # 输入数据名称列表
    output_data_names: list = []  # 输出数据名称列表

    def __init__(self, n_outputs):
        SQLiteCache.__init__(self, db_path="./pipeline.db")
        Stage.__init__(self)
        self._job_id = None
        self._stage_idx = None
        self._n_outputs = n_outputs
        self._collect_result = False     # forward函数之后对LazyFrame是否执行collect
        self._show_collect = False
        self.logger = logging
        self.summary = []

    def set_job_id(self, job_id):
        self._job_id = job_id

    def get_run_folder(self):
        if self._job_id is None:
            raise ValueError(
                "job_id has not been set. Please ensure set_job_id is called before running the stage."
            )
        return os.path.join(LARGE_DATA_PATH, self._job_id)

    def _get_data_path(self, name):
        """生成用于存储数据的文件路径"""
        hash_name = hashlib.md5(name.encode()).hexdigest()
        return os.path.join(self.get_run_folder(), f"{name}_{hash_name}")

    async def read(self, k: str) -> object:
        """从sqlite获取文件位置并读取"""
        file_path = await SQLiteCache.read(self, k)
        self.logger.info(f"获取文件位置, 开始读取: {file_path}")
        if file_path and os.path.exists(file_path):
            # async with aiofiles.open(file_path, "rb") as f:
            #     data = await f.read()
            #     data_type = 'pandas' if file_path.endswith('.pandas') else 'polars' if file_path.endswith('.polars') else 'pickle'
            #     try:
            #         return deserialize(data, data_type)
            #     except Exception as e:
            #         self.logger.error(f"反序列化失败: {e}")
            #         raise e

            try:
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()
                    if file_path.endswith('.parquet'):
                        data = pl.scan_parquet(data)
                    elif file_path.endswith('.pickle'):
                        data = pickle.loads(data)
                self.logger.info("读取成功")
                return data
            except Exception as e:
                self.logger.error(f"读取数据失败: {e}")
                raise e
        else:
            self.logger.warning(f"数据文件不存在: {file_path}")
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

    async def write(self, k: str, v: object) -> None:
        """写入输出数据到本地磁盘，并记录文件位置到sqlite"""
        file_path = self._get_data_path(k)
        data_type = 'parquet' if isinstance(v, (pl.DataFrame, pl.LazyFrame)) else 'pickle'
        file_path += f".{data_type}"
        self.logger.info(f"数据{k}开始写入: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # async with aiofiles.open(file_path, "wb") as f:
        #     try:
        #         data = serialize(v)
        #         await f.write(data)
        #     except Exception as e:
        #         self.logger.error(f"序列化失败: {e}")
        #         raise e

        try:
            if isinstance(v, pd.DataFrame):
                raise TypeError("请使用polars替代pandas实现")
            elif isinstance(v, pl.DataFrame):
                v.write_parquet(file_path)
            elif isinstance(v, pl.LazyFrame):
                v.sink_parquet(file_path)
            else:
                async with aiofiles.open(file_path, "wb") as f:
                    pickle.dump(v, f)
            
            await SQLiteCache.write(self, k, file_path)
            self.logger.info(f"数据{k}已写入文件: {file_path}")

        except Exception as e:
            self.logger.error(f"写入文件失败: {e}")
            raise e

    @property
    def name(self) -> str:
        """
        返回一个唯一的名称，由类名和实例的UUID组成。
        """
        if self._name:
            return self._name
        return f"{self._stage_idx}_{self.__class__.__name__}"
    
    @name.setter
    def name(self, value: str):
        self._name = value

    def set_input(self, input_data_name: str):
        """
        设置单个输入数据名称。

        参数:
            input_data_name (str 或 list): 输入数据名称。

        返回:
            self: 返回实例本身，支持链式调用。

        异常:
            TypeError: 如果输入类型不匹配。
        """
        if isinstance(input_data_name, list) and len(input_data_name) == 1:
            self.input_data_names = input_data_name
        elif isinstance(input_data_name, str):
            self.input_data_names = [input_data_name]
        else:
            raise TypeError("input_data_name类型不匹配")
        self.logger.info(f"Stage: {self.name} set input data: {self.input_data_names}")
        return self

    def add_input(self, input_data_name: str):
        """
        添加一个输入数据名称。

        参数:
            input_data_name (str): 要添加的输入数据名称。

        返回:
            self: 返回实例本身，支持链式调用。
        """
        if self.input_data_names:
            self.input_data_names.append(input_data_name)
        else:
            self.set_input(input_data_name)
        return self

    def set_inputs(self, input_data_names: list[str]):
        """
        设置多个输入数据名称。

        参数:
            input_data_names (list[str]): 输入数据名称列表。

        返回:
            self: 返回实例本身，支持链式调用。

        异常:
            AssertionError: 如果输入不是列表类型。
        """
        assert isinstance(input_data_names, list)
        self.input_data_names = input_data_names
        self.logger.info(f"Stage: {self.name} set input data: {self.input_data_names}")
        return self

    def set_n_outputs(self):
        """
        设置默认输出数据名称。

        参数:
            n_outputs (int): 输出数量。
            force_new (bool): 是否强制生成新的输出名称。

        返回:
            self: 返回实例本身，支持链式调用。
        """
        self.output_data_names = [
            f"{self.name}_output_{i}" for i in range(self._n_outputs)
        ]
        self.logger.info(f"Stage: {self.name} set output data: {self.output_data_names}")
        return self
    
    def get_output_names(self):
        return self.output_data_names

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self._job_id = pipeline.job_id
        self._stage_idx = pipeline.get_cur_stage_idx()
        self.set_n_outputs()
        pipeline.stages_to_add.append(self)
        return self
    
    def collect_result(self, show: bool = False):
        self._collect_result = True
        self._show_collect = show
        return self

    def forward(self, *args, **kwargs):
        """
        前向传播方法，需要在子类中实现。

        异常:
            NotImplementedError: 如果子类没有实现此方法。
        """
        raise NotImplementedError()

    async def run(self, *args, **kwargs):
        """
        运行阶段的主要方法。
        读取输入数据，调用forward方法，并写入输出数据。

        参数:
            *args: 可变位置参数。
            **kwargs: 可变关键字参数。
        """

        if self.input_data_names:
            input_datas = []
            self.logger.warning(f"stage: {self.name} 开始读取输入: {self.input_data_names}")
            for name in tqdm(self.input_data_names, desc="Reading input data"):
                try:
                    data = await self.read(name)
                    if data is None:
                        raise ValueError(f"无法读取输入数据: {name}")
                    input_datas.append(data)
                except Exception as e:
                    self.logger.error(f"Stage {self.__class__.__name__} 读取输入数据 {name} 时出错: {e}")
                    raise RuntimeError(e)
            self.logger.warning(f"stage: {self.name} 开始运行")
            # 运行
            outs = self.forward(*input_datas, *args, **kwargs)
        else:
            self.logger.warning(f"stage: {self.name} 无任何输入")
            self.logger.warning(f"stage: {self.name} 开始运行")
            # 运行
            outs = self.forward(*args, **kwargs)

        if self.output_data_names:

            if len(self.output_data_names) == 1:
                o_n = self.output_data_names[0]
                # if self._collect_result and isinstance(outs, pl.LazyFrame):
                if isinstance(outs, pl.LazyFrame):
                    outs = outs.collect()
                if self._show_collect:
                    self.logger.info(f"[Show Collect Result of Output {o_n}]\n{outs}")
                await self.write(o_n, outs)
            else:
                for o_n, o in zip(self.output_data_names, outs):
                    # print(o.collect())
                    # if self._collect_result and isinstance(o, pl.LazyFrame):
                    if isinstance(o, pl.LazyFrame):
                        o = o.collect()
                    if self._show_collect:
                        self.logger.info(f"[Show Collect Result of Output {o_n}]\n{o}")
                    await self.write(o_n, o)
        else:
            self.logger.warning(f"stage: {self.name} 无任何输出")

        if len(self.summary) > 0:
            """summary需要长这样: [{"图表1": 图表.dump_options_with_quotes()}, {"图表2": 图表.dump_options_with_quotes()}]"""
            # if not isinstance(self.summary, list) or not all(isinstance(item, dict) for item in self.summary):
            #     raise TypeError("summary需要是包含字典的列表,每个字典包含图表名称和对应的dump_options_with_quotes()结果!")
            self.logger.warning(f"stage: {self.name} 存在summary，开始写入数据库")
            await SQLiteCache.write(self, f"{self._job_id}_{self.name}_summary", pickle.dumps(self.summary))


class CustomStage(BaseStage):
    """
    基础阶段类，继承自CustomStage。
    提供了设置输入输出、运行阶段等基本功能。
    """

    pass


class DecoratorStage(BaseStage):
    """
    Class to wrap any user-defined function decorated with the stage decorator.

    stage_function: The callable defined by the user-defined function pipeline
                    stage.
    args: The arguments to the user-defined function pipeline stage.
    kwargs: The keyword arguments to the user-defined function pipeline stage.
    """

    def __init__(self, stage_function: callable, n_outputs: int, *args, **kwargs):
        super().__init__(n_outputs=n_outputs)

        self.stage_function = stage_function
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Name is given by the name of the user-defined decorated function."""
        if self._name:
            return self._name
        return f"{self._stage_idx}_{self.stage_function.__name__}"
    
    @name.setter
    def name(self, value: str):
        self._name = value

    def forward(self, *args, **kwargs) -> None:
        """
        Stage is run by calling the wrapped user function with its arguments.
        """
        if self.args:
            args = args + self.args
        if self.kwargs:
            kwargs.update(self.kwargs)
        return self.stage_function(*args, **kwargs)


class StageExecutor(PickleSerializer, SQLiteCache):
    """
    Context manager for the execution of a stage, or group of stages, of a
    pipeline.

    The setup phase (__enter__) persists relevant metadata such as the stages
    currently in progress to a SQLite database.

    The teardown phase (__exit__) deletes relevant metadata from the SQLite
    database.

    db_path: Path to the SQLite database.
    stages: The stages that are currently in progress.
    """

    async def __init__(self, db_path, stages):
        SQLiteCache.__init__(self, db_path)

        self.pipeline = await self.read("pipeline")
        self.stages = stages

    async def __enter__(self):
        await self.write("in_progress", self.serialize(self.stages))
        return self

    async def __exit__(self, exc_type, exc_val, exc_tb):
        completed = self.deserialize(await self.read("in_progress"))
        current_done = await self.read("done")
        if current_done is None:
            current_done = []
        else:
            current_done = self.deserialize(current_done)
        current_done += completed
        await self.write("done", self.serialize(current_done))
        await self.delete("in_progress")

    @staticmethod
    def execute(fn: callable, *args, **kwargs) -> None:
        """Execute the stage/group of stages."""
        fn(*args, **kwargs)


def stage(n_outputs: int):
    def decorator(stage_function: callable):
        def wrapper(*args, **kwargs) -> DecoratorStage:
            return DecoratorStage(stage_function, n_outputs, *args, **kwargs)

        return wrapper

    return decorator
