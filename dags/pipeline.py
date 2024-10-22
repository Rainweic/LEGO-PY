"""
此模块包含'Pipeline'的实现:与DAG同义的实体,包含一个或多个'Stages'。
pipeline将相关的DAG建模为networkx.DiGraph对象,并确保在添加节点/阶段时DAG属性保持满足。

pipeline的执行可以在串行模式或并行模式下运行。在串行模式下,阶段按正确的顺序执行
(由阶段之间的相互依赖关系定义),但按顺序运行。在此模式下,可以并行运行的阶段不会并行。
在并行模式下,可以并行运行的阶段会并行运行,并在pipeline执行时指定的核心数量之间分配。

pipeline中可以并行运行的一组阶段称为"Group"。在串行模式下,Groups只包含当前正在执行的单个阶段。
在并行模式下,Groups包含当时可以并行运行的所有节点。在这两种情况下,组都在定制的上下文管理器
(pydags.stage.StageExecutor)中运行。在此上下文管理器的启动和拆卸期间,
与DAG执行相关的相关信息(如正在进行的阶段、已完成的阶段等)被写入SQLite。
然后可以从SQLite读取此信息并在下游使用。
"""

import os
import pickle
import typing
import logging
import datetime
import hashlib
import json
import asyncio
import aiofiles
import enum

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from io import BytesIO
from multiprocessing.pool import ThreadPool

from .serialization import CloudPickleSerializer
from .cache import SQLiteCache
from .stage import StageExecutor, BaseStage
from utils.logger import setup_logger


class StageException(Exception):
    pass


class InvalidStageTypeException(StageException):
    pass


class DAGException(Exception):
    pass


class DAGVerificationException(DAGException):
    pass


class StageStatus(enum.Enum):
    DEFAULT = "default"
    WAITING = "waiting"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class Pipeline(CloudPickleSerializer, SQLiteCache):
    """
    pydags的Pipeline功能的实现。Pipeline的目标是协调DAG组成阶段的执行,
    并在执行期间写入相关的Stage元数据。目前,此元数据包括正在进行的节点和已完成的节点。

    pipeline: networkx.DiGraph的实例,用于容纳pipeline的底层DAG。
              使用此库的原因是因为已经实现了各种重要的功能,
              包括拓扑排序、图形可视化、众多图形算法等。
    """

    def __init__(self, parallel: int = None, visualize: bool = False, save_dags: bool = True, force_rerun: bool = False, job_id=None):
        """
        我们使用SQLite数据库作为Pipeline的底层缓存。
        """
        self.pipeline = nx.DiGraph()
        self.db_path = "pipeline.db"
        SQLiteCache.__init__(self, self.db_path)
        if job_id:
            self.job_id = job_id
        else:
            self.job_id = self.generate_job_id()
        self.logger = setup_logger("pipeline", "PIPELINE", job_id=self.job_id)
        self.stage_counter = 0
        self.stages_to_add = list()
        self.completed_stages = list()

        self.parallel = parallel
        self.visualize = visualize
        self.save_dags = save_dags
        self.force_rerun = force_rerun

        self._initialize_stage_statuses()

    def generate_job_id(self):
        self.job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.job_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 在退出上下文时可以添加一些清理操作
        
        for stage in self.stages_to_add:
            self.add_stage(stage=stage)
        
        await self.start(parallel=self.parallel,
                         visualize=self.visualize,
                         save_dags=self.save_dags,
                         force_rerun=self.force_rerun)

    def topological_sort_grouped(self) -> typing.Generator:
        """
        对DAG/pipeline执行拓扑排序的方法。但是,此排序与nx.topological_sort不同,因为它是分组的。
        这意味着DAG每个级别上可以并行运行的阶段被分组在一起。这是为了在用户希望跨核心分配pipeline
        执行的情况下进行的。在这种情况下,每组中的阶段将并行运行。但默认行为是串行运行整个pipeline,
        包括同一可并行化组内的阶段。

        返回:
             生成器,其中每个元素是同一组中的节点列表。
        """

        self.logger.info("计算pipeline DAG的分组拓扑排序")
        indegree_map = {v: d for v, d in self.pipeline.in_degree() if d > 0}
        zero_indegree = [v for v, d in self.pipeline.in_degree() if d == 0]
        while zero_indegree:
            yield zero_indegree
            new_zero_indegree = []
            for v in zero_indegree:
                for _, child in self.pipeline.edges(v):
                    indegree_map[child] -= 1
                    if not indegree_map[child]:
                        new_zero_indegree.append(child)
            zero_indegree = new_zero_indegree

    def get_cur_stage_idx(self):
        _idx = self.stage_counter
        self.stage_counter += 1
        return _idx

    def add_stage(self, stage: BaseStage) -> None:
        """
        向pipeline添加阶段的方法。如果阶段已经存在于DAG中,则不会添加
        (尽管networkx可以处理这种情况)。阶段根据其名称(通常是用户定义的类或函数名称)
        和一个名为'stage_wrapper'的属性定义,该属性是BaseStage子类的实际实例。
        此对象用于运行相关的DAG阶段。

        此外,我们在DAG中添加阶段与其前置阶段(由用户定义)之间的边。

        最后,进行检查以确保在添加阶段后,DAG仍然确实是一个DAG。
        """

        if not isinstance(stage, BaseStage):
            raise InvalidStageTypeException(
                "请确保您的阶段是pydags.stage.BaseStage的子类"
            )

        self.pipeline.add_node(stage.name, stage_wrapper=stage)

        for preceding_stage in stage.preceding_stages:
            self.pipeline.add_edges_from([(preceding_stage.name, stage.name)])

        if not nx.is_directed_acyclic_graph(self.pipeline):
            raise DAGVerificationException("Pipeline不再是一个DAG!")

    def _compute_pipeline_hash(self):
        """
        计算当前pipeline的hash值，用于检查pipeline是否发生改动
        """
        nodes = list(self.pipeline.nodes)
        edges = list(self.pipeline.edges)
        pipeline_repr = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True)
        return hashlib.md5(pipeline_repr.encode()).hexdigest()

    async def _save_checkpoint(self):
        """
        保存检查点到SQLite数据库。
        """
        self.logger.info("save checkpoint")
        checkpoint_data = json.dumps(
            {
                "completed_stages": self.completed_stages,
                "pipeline_hash": self._compute_pipeline_hash(),
            }
        )
        await self.write("pipeline_checkpoint", checkpoint_data)

    async def _load_checkpoint(self):
        """
        从SQLite数据库加载检查点。
        """
        checkpoint = await self.read("pipeline_checkpoint")
        if checkpoint:
            data = json.loads(checkpoint)
            self.completed_stages = data["completed_stages"]
            return data
        return None

    async def _delete_checkpoint(self):
        """
        删除检查点。
        """
        self.logger.info("删除检查点。")
        await self.delete("pipeline_checkpoint")

    def _initialize_stage_statuses(self):
        """
        初始化所有stage的状态为DEFAULT
        """
        for stage_name in self.pipeline.nodes:
            self.write_sync(f"{stage_name}", StageStatus.DEFAULT.value)

    async def _update_stage_status(self, stage_name: str, status: StageStatus):
        """
        更新stage的状态
        """
        await self.write(f"{self.job_id}_{stage_name}", status.value)

    async def _get_stage_status(self, stage_name: str):
        """
        获取stage的状态
        """
        status = await self.read(f"stage_status_{self.job_id}_{stage_name}")
        return StageStatus(status) if status else StageStatus.DEFAULT
    
    async def _write_output_names(self, stage_name):
        output_names = self.pipeline.nodes[stage_name]['stage_wrapper'].get_output_names()
        self.logger.info(f"Save output names to sqlite: {output_names}")
        await self.write(f"{self.job_id}_{stage_name}_output_names", pickle.dumps(output_names))

    async def run_stage(self, stage_name: str) -> None:
        """
        运行pipeline/DAG特定阶段的方法。使用阶段名称获取BaseStage的相关实例,
        并使用所需的'run'方法执行。

        参数:
            stage_name <str>: pipeline中阶段的名称。
        """
        self.logger.info(f"[Running stage] {stage_name}")
        await self._update_stage_status(stage_name, StageStatus.RUNNING)
        try:
            await self.pipeline.nodes[stage_name]["stage_wrapper"].run()
            await self._update_stage_status(stage_name, StageStatus.SUCCESS)
            await self._write_output_names(stage_name)
            self.completed_stages.append(stage_name)
        except Exception as e:
            self.logger.error(f"Stage {stage_name} failed: {str(e)}")
            await self._update_stage_status(stage_name, StageStatus.FAILED)
            raise e
        finally:
            await self._save_checkpoint()

    # 主要函数入口
    async def start(
        self,
        parallel: bool = True,
        visualize: bool = False,
        save_dags: bool = True,
        force_rerun: bool = False,
    ) -> None:
        """
        执行pipeline(及其所有组成阶段)的方法。执行顺序由分组拓扑排序定义。
        """

        if visualize:
            await self._visualize(save_dags)

        self.logger.info("序列化pipeline并写入SQLite")
        await self.write("pipeline", self.serialize(self.pipeline))

        if force_rerun:
            self.logger.info("强制重跑所有阶段")
            self.completed_stages = list()
            await self._delete_checkpoint()
            self.logger.info(f"所有节点: {self.pipeline.nodes}")
        else:
            checkpoint_data = await self._load_checkpoint()
            if checkpoint_data:
                previous_pipeline_hash = checkpoint_data["pipeline_hash"]
                self.logger.info(f"从检查点恢复, 已完成的阶段: {self.completed_stages}")
            else:
                previous_pipeline_hash = None

            current_pipeline_hash = self._compute_pipeline_hash()

            if previous_pipeline_hash and previous_pipeline_hash != current_pipeline_hash:
                self.logger.info("Pipeline 发生改动，重头运行")
                self.completed_stages = list()

        sorted_grouped_stages = self.topological_sort_grouped()
        for group in sorted_grouped_stages:
            stages_to_run = [stage for stage in group if stage not in self.completed_stages]
            self.logger.info("处理组: %s", stages_to_run)

            if parallel:
                tasks = [self.run_stage(stage) for stage in stages_to_run]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for stage, result in zip(stages_to_run, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Stage {stage} failed: {str(result)}")
            else:
                for stage in stages_to_run:
                    await self.run_stage(stage)
                        # break

        await self._delete_checkpoint()
        self.logger.info("Finish all tasks.")

    async def get_graph_last_output(self):
        return await self.read(self.completed_stages[-1])
    
    async def get_output(self, output_name):
        file_path = await self.read(output_name)
        if file_path and os.path.exists(file_path):
            async with aiofiles.open(file_path, "rb") as f:
                return pickle.loads(await f.read())
        else:
            self.logger.warning(f"数据文件不存在: {file_path}")
            return None

    async def _visualize(self, save_dags: bool = False):
        """
        异步地通过渲染matplotlib图形来可视化pipeline/DAG的方法，并可选择保存到本地。
        """
        drawing = nx.drawing.nx_pydot.to_pydot(self.pipeline)
        png_str = drawing.create_png()
        sio = BytesIO()
        sio.write(png_str)
        sio.seek(0)

        img = mpimg.imread(sio)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")

        if save_dags:
            save_path = os.path.join("./visualizations", f"pipeline_{self.job_id}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
            print(f"Pipeline visualization saved to: {save_path}")
        else:
            plt.show()