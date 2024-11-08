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
import datetime
import hashlib
import json
import aiofiles
import enum

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from io import BytesIO

from .serialization import CloudPickleSerializer
from .cache import SQLiteCache
from .stage import BaseStage
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

        self.stages = []  # 用列表按顺序存储阶段
        self.stage_dict = {}  # 用字典存储阶段，键为阶段名称
        self.dependencies = {}  # 存储阶段间的依赖关系

    def generate_job_id(self):
        self.job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.job_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 在退出上下文时可以添加一些清理操作
        
        for stage in self.stages_to_add:
            self.add_stage(stage=stage)
        
        await self.start(visualize=self.visualize,
                         save_dags=self.save_dags,
                         force_rerun=self.force_rerun)

    def get_cur_stage_idx(self):
        _idx = self.stage_counter
        self.stage_counter += 1
        return _idx

    async def _save_checkpoint(self):
        """
        保存检查点到SQLite数据库。
        """
        self.logger.info("save checkpoint")
        checkpoint_data = json.dumps(
            {
                "completed_stages": self.completed_stages,
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
            self.write_sync(f"{stage_name}", StageStatus.WAITING.value)

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

    def _build_dependencies(self):
        for stage in self.stages:
            for preceding_stage in stage.preceding_stages:
                self.dependencies[stage.name].add(preceding_stage.name)

    def _topological_sort(self):
        in_degree = {stage.name: len(self.dependencies[stage.name]) for stage in self.stages}
        queue = [stage.name for stage in self.stages if in_degree[stage.name] == 0]
        sorted_stages = []

        while queue:
            current_stage = queue.pop(0)
            sorted_stages.append(current_stage)

            for stage in self.stages:
                if current_stage in self.dependencies[stage.name]:
                    in_degree[stage.name] -= 1
                    if in_degree[stage.name] == 0:
                        queue.append(stage.name)

        if len(sorted_stages) != len(self.stages):
            raise DAGVerificationException("Pipeline中存在循环依赖!")

        return sorted_stages

    def _is_acyclic(self):
        return len(self._topological_sort()) == len(self.stages)
    
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

        self.logger.debug(f"add_stage {stage.name}")
        self.pipeline.add_node(stage.name, stage_wrapper=stage)

        for preceding_stage in stage.preceding_stages:
            self.pipeline.add_edges_from([(preceding_stage.name, stage.name)])

        if not nx.is_directed_acyclic_graph(self.pipeline):
            raise DAGVerificationException("Pipeline不再是一个DAG!")

        self.stages.append(stage)
        self.stage_dict[stage.name] = stage
        self.dependencies[stage.name] = set()
    
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
            await self.stage_dict[stage_name].run()  # 使用 stage_dict 而不是 stages
            await self._update_stage_status(stage_name, StageStatus.SUCCESS)
            await self._write_output_names(stage_name)
            self.completed_stages.append(stage_name)
        except Exception as e:

            await self._update_stage_status(stage_name, StageStatus.FAILED)

            self.stage_dict[stage_name].logger.error(f"\n------ 报错 job_id: {self.job_id} ------")
            error_msg = f"Stage {stage_name} failed: {str(e)}"
            self.stage_dict[stage_name].logger.error(error_msg)
            self.stage_dict[stage_name].logger.exception("\n------ 完整的错误栈信息 ------\n")
            self.stage_dict[stage_name].logger.error(f"\n------ 报错 job_id: {self.job_id} ------")

            # raise e
        finally:
            await self._save_checkpoint()

    # 主要函数入口
    async def start(
        self,
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
        await self.write("pipeline", self.serialize(self.stage_dict))

        if force_rerun:
            self.logger.info("强制重跑所有阶段")
            self.completed_stages = list()
            await self._delete_checkpoint()
            # TODO 运行全部 会有遗漏最后一个节点
        else:
            checkpoint_data = await self._load_checkpoint()
            if checkpoint_data:
                self.logger.info(f"从检查点恢复, 已完成的阶段: {self.completed_stages}")

        sorted_stages = self._topological_sort()
        self.logger.info(f"所有节点: {sorted_stages}, 跳过的节点[已完成]: {self.completed_stages}")
        for stage_name in sorted_stages:
            if stage_name not in self.completed_stages:
                await self.run_stage(stage_name)

        await self._delete_checkpoint()
        self.logger.info("Finish all tasks.")



