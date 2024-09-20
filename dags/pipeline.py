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

from io import BytesIO
from multiprocessing.pool import ThreadPool
import typing
import logging

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .serialization import CloudPickleSerializer
from .cache import SQLiteCache
from .stage import StageExecutor, Stage


class StageException(Exception):
    pass


class InvalidStageTypeException(StageException):
    pass


class DAGException(Exception):
    pass


class DAGVerificationException(DAGException):
    pass


class Pipeline(CloudPickleSerializer, SQLiteCache):
    """
    pydags的Pipeline功能的实现。Pipeline的目标是协调DAG组成阶段的执行,
    并在执行期间写入相关的Stage元数据。目前,此元数据包括正在进行的节点和已完成的节点。

    pipeline: networkx.DiGraph的实例,用于容纳pipeline的底层DAG。
              使用此库的原因是因为已经实现了各种重要的功能,
              包括拓扑排序、图形可视化、众多图形算法等。
    """

    pipeline = nx.DiGraph()

    def __init__(self):
        """
        我们使用SQLite数据库作为Pipeline的底层缓存。
        """
        self.db_path = 'pipeline.db'
        SQLiteCache.__init__(self, self.db_path)

    def topological_sort_grouped(self) -> typing.Generator:
        """
        对DAG/pipeline执行拓扑排序的方法。但是,此排序与nx.topological_sort不同,因为它是分组的。
        这意味着DAG每个级别上可以并行运行的阶段被分组在一起。这是为了在用户希望跨核心分配pipeline
        执行的情况下进行的。在这种情况下,每个组中的阶段将并行运行。但默认行为是串行运行整个pipeline,
        包括同一可并行化组内的阶段。

        返回:
             生成器,其中每个元素是同一组中的节点列表。
        """

        logging.info('计算pipeline DAG的分组拓扑排序')
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

    def add_stage(self, stage: Stage) -> None:
        """
        向pipeline添加阶段的方法。如果阶段已经存在于DAG中,则不会添加
        (尽管networkx可以处理这种情况)。阶段根据其名称(通常是用户定义的类或函数名称)
        和一个名为'stage_wrapper'的属性定义,该属性是BaseStage子类的实际实例。
        此对象用于运行相关的DAG阶段。

        此外,我们在DAG中添加阶段与其前置阶段(由用户定义)之间的边。

        最后,进行检查以确保在添加阶段后,DAG仍然确实是一个DAG。
        """

        if not isinstance(stage, Stage):
            raise InvalidStageTypeException('请确保您的阶段是pydags.stage.Stage的子类')

        self.pipeline.add_node(stage.name, stage_wrapper=stage)

        for preceding_stage in stage.preceding_stages:
            self.pipeline.add_edges_from([(preceding_stage.name, stage.name)])

        if not nx.is_directed_acyclic_graph(self.pipeline):
            raise DAGVerificationException('Pipeline不再是一个DAG!')

    def add_stages(self, stages: typing.List[Stage]) -> None:
        """
        向pipeline添加阶段列表的方法。如果段已经存在于DAG中,则不会添加
        (尽管networkx可以处理这种情况)。阶段根据其名称(通常是用户定义的类或函数名称)
        和一个名为'stage_wrapper'的属性定义,该属性是BaseStage子类的实际实例。
        此对象用于运行相关的DAG阶段。

        此外,我们在DAG中添加阶段与其前置阶段(由用户定义)之间的边。

        最后,进行检查以确保在添加阶段后,DAG仍然确实是一个DAG。
        """

        for stage in stages:
            self.add_stage(stage)

    def run_stage(self, stage_name: str) -> None:
        """
        运行pipeline/DAG特定阶段的方法。使用阶段名称获取BaseStage的相关实例,
        并使用所需的'run'方法执行。

        参数:
            stage_name <str>: pipeline中阶段的名称。
        """
        self.pipeline.nodes[stage_name]['stage_wrapper'].run()

    def start(self, num_cores: int = None, visualize: bool = False) -> None:
        """
        执行pipeline(及其所有组成阶段)的方法。执行顺序由分组拓扑排序定义。
        如果num_cores是正整数,则组内的阶段将并行执行(跨核心)。
        如果num_cores保持为None(默认情况),则整个pipeline(包括同一组内的阶段)将串行运行。

        参数:
            num_cores [<int>, <None>]: 要分配的核心数。
        """

        if visualize:
            self.visualize()

        logging.info('序列化pipeline并写入SQLite')
        self.write('pipeline', self.serialize(self.pipeline))

        sorted_grouped_stages = self.topological_sort_grouped()
        for group in sorted_grouped_stages:
            logging.info('处理组: %s', group)
            if num_cores:
                # TODO 并行时read可能会取串了
                pool = ThreadPool(num_cores)
                with StageExecutor(self.db_path, group) as stage_executor:
                    stage_executor.execute(pool.map, self.run_stage, group)
            else:
                for stage in group:
                    with StageExecutor(self.db_path, [stage]) as stage_executor:
                        stage_executor.execute(self.run_stage, stage)
        self.delete('done')

    def visualize(self):
        """
        通过渲染matplotlib图形来可视化pipeline/DAG的方法。

        参考:
            https://stackoverflow.com/questions/10379448/plotting-directed-graphs-in-python-in-a-way-that-show-all-edges-separately
        """

        drawing = nx.drawing.nx_pydot.to_pydot(self.pipeline)

        png_str = drawing.create_png()
        sio = BytesIO()
        sio.write(png_str)
        sio.seek(0)

        img = mpimg.imread(sio)
        plt.imshow(img)

        plt.show()
