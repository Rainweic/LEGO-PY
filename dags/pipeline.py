"""
Pipeline模块: 实现DAG(有向无环图)的执行流程控制

主要功能:
1. 管理和执行DAG中的各个Stage
2. 提供串行/并行执行模式
3. 处理Stage间的依赖关系
4. 提供执行状态管理和可视化
"""

# 标准库导入
import os
import json
import enum
import pickle
import datetime
from io import BytesIO

# 第三方库导入
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 本地导入
from .serialization import CloudPickleSerializer
from .cache import SQLiteCache
from .stage import BaseStage
from utils.logger import setup_logger

# 异常类定义
class StageException(Exception):
    pass

class InvalidStageTypeException(StageException):
    pass

class DAGException(Exception):
    pass

class DAGVerificationException(DAGException):
    pass

# 状态枚举
class StageStatus(enum.Enum):
    DEFAULT = "default"
    WAITING = "waiting"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"

class Pipeline(CloudPickleSerializer, SQLiteCache):
    """Pipeline类: 管理和执行DAG流程"""

    def __init__(self, parallel: int = None, visualize: bool = False, 
                 save_dags: bool = True, force_rerun: bool = False, job_id=None):
        """初始化Pipeline"""
        # 基础设置
        self.pipeline = nx.DiGraph()
        self.db_path = "pipeline.db"
        SQLiteCache.__init__(self, self.db_path)
        
        # 作业标识
        self.job_id = job_id if job_id else self.generate_job_id()
        self.logger = setup_logger("pipeline", "PIPELINE", job_id=self.job_id)
        self.process_id = os.getpid()
        
        # 状态追踪
        self.stage_counter = 0
        self.stages_to_add = list()
        self.completed_stages = list()
        
        # 执行配置
        self.parallel = parallel
        self.visualize = visualize
        self.save_dags = save_dags
        self.force_rerun = force_rerun
        
        # Stage管理
        self.stages = []  # 顺序存储
        self.stage_dict = {}  # 名称索引
        self.dependencies = {}  # 依赖关系
        
        # 初始化
        self._initialize_stage_statuses()

    # ====== 生命周期管理 ======
    def generate_job_id(self):
        """生成唯一的作业ID"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文退出时执行清理和启动"""
        for stage in self.stages_to_add:
            self.add_stage(stage=stage)
        
        await self.start(visualize=self.visualize,
                        save_dags=self.save_dags,
                        force_rerun=self.force_rerun)

    # ====== 检查点管理 ======
    async def _save_checkpoint(self):
        """保存执行检查点"""
        checkpoint_data = json.dumps({"completed_stages": self.completed_stages})
        await self.write("pipeline_checkpoint", checkpoint_data)
        self.logger.info("检查点已保存")

    async def _load_checkpoint(self):
        """加载执行检查点"""
        checkpoint = await self.read("pipeline_checkpoint")
        if checkpoint:
            data = json.loads(checkpoint)
            self.completed_stages = data["completed_stages"]
            return data
        return None

    async def _delete_checkpoint(self):
        """删除检查点"""
        await self.delete("pipeline_checkpoint")
        self.logger.info("检查点已删除")

    # ====== 状态管理 ======
    def _initialize_stage_statuses(self):
        """初始化所有Stage状态"""
        # for stage_name in self.pipeline.nodes:
        #     self.write_sync(f"{stage_name}", StageStatus.WAITING.value)
        self._update_stage_statuses(StageStatus.DEFAULT)

    def _update_stage_statuses(self, status: StageStatus):
        for stage_name in self.pipeline.nodes:
            self.write_sync(f"{stage_name}", status.value)

    async def _update_stage_status(self, stage_name: str, status: StageStatus):
        """更新Stage状态"""
        await self.write(f"{self.job_id}_{stage_name}", status.value)

    async def _get_stage_status(self, stage_name: str):
        """获取Stage状态"""
        status = await self.read(f"stage_status_{self.job_id}_{stage_name}")
        return StageStatus(status) if status else StageStatus.DEFAULT

    # ====== Stage管理 ======
    def add_stage(self, stage: BaseStage) -> None:
        """添加新的Stage到Pipeline"""
        if not isinstance(stage, BaseStage):
            raise InvalidStageTypeException("Stage必须继承自BaseStage")

        self.logger.debug(f"添加Stage: {stage.name}")
        self.pipeline.add_node(stage.name, stage_wrapper=stage)

        # 添加依赖关系
        for preceding_stage in stage.preceding_stages:
            self.pipeline.add_edges_from([(preceding_stage.name, stage.name)])

        # 验证DAG
        if not nx.is_directed_acyclic_graph(self.pipeline):
            raise DAGVerificationException("添加Stage后Pipeline不再是DAG!")

        self.stages.append(stage)
        self.stage_dict[stage.name] = stage
        self.dependencies[stage.name] = set()

    async def run_stage(self, stage_name: str) -> None:
        """执行单个Stage"""
        self.logger.info(f"[执行Stage] {stage_name}")
        await self._update_stage_status(stage_name, StageStatus.RUNNING)
        
        try:
            await self.stage_dict[stage_name].run()
            await self._update_stage_status(stage_name, StageStatus.SUCCESS)
            await self._write_output_names(stage_name)
            self.completed_stages.append(stage_name)
            
        except Exception as e:
            await self._update_stage_status(stage_name, StageStatus.FAILED)
            self._log_error(stage_name, e)
            
        finally:
            await self._save_checkpoint()

    # ====== 辅助方法 ======
    def get_cur_stage_idx(self):
        """获取当前Stage索引"""
        _idx = self.stage_counter
        self.stage_counter += 1
        return _idx

    def _log_error(self, stage_name: str, error: Exception):
        """记录错误日志"""
        logger = self.stage_dict[stage_name].logger
        logger.error(f"\n------ 报错 job_id: {self.job_id} ------")
        logger.error(f"Stage {stage_name} failed: {str(error)}")
        logger.exception("\n------ 完整的错误栈信息 ------\n")
        logger.error(f"\n------ 报错 job_id: {self.job_id} ------")

    async def _write_output_names(self, stage_name):
        """保存Stage输出名称"""
        output_names = self.pipeline.nodes[stage_name]['stage_wrapper'].get_output_names()
        self.logger.info(f"保存输出名称: {output_names}")
        await self.write(f"{self.job_id}_{stage_name}_output_names", pickle.dumps(output_names))

    # ====== 可视化 ======
    async def _visualize(self, save_dags: bool = False):
        """可视化Pipeline"""
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
            self.logger.info(f"Pipeline可视化已保存: {save_path}")
        else:
            plt.show()

    # ====== 拓扑排序 ======
    def _topological_sort(self):
        """对Stage进行拓扑排序"""
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

    # ====== 主要入口 ======
    async def start(self, visualize: bool = False, save_dags: bool = True, 
                   force_rerun: bool = False) -> None:
        """启动Pipeline执行"""
        if visualize:
            await self._visualize(save_dags)

        await self.write(f"pipeline_{self.job_id}", self.serialize(self))

        self.logger.info("开始Pipeline执行")

        # 处理重跑逻辑
        if force_rerun:
            self.logger.info("强制重跑所有阶段")
            self.completed_stages = list()
            await self._delete_checkpoint()
        else:
            checkpoint_data = await self._load_checkpoint()
            if checkpoint_data:
                self.logger.info(f"从检查点恢复, 已完成的阶段: {self.completed_stages}")

        # 执行Stages
        sorted_stages = self._topological_sort()
        self.logger.info(f"所有节点: {sorted_stages}, 跳过的节点[已完成]: {self.completed_stages}")
        
        for stage_name in sorted_stages:
            if stage_name not in self.completed_stages:
                await self.run_stage(stage_name)

        await self._delete_checkpoint()
        self.logger.info("Pipeline执行完成")



