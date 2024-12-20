import polars as pl
from dags.stage import CustomStage


class Where(CustomStage):
    """
    Where 类用于根据条件过滤 LazyFrame。

    这个类继承自 CustomStage，用于在数据处理管道中执行数据过滤操作。
    它可以根据指定的条件来过滤 LazyFrame 中的行。

    属性:
        conditions (tuple): 要应用的过滤条件。

    方法:
        forward(lf: pl.LazyFrame) -> pl.LazyFrame: 执行过滤操作并返回过滤后的 LazyFrame。
    """

    def __init__(self, conditions: list[str]):
        super().__init__(n_outputs=1)
        self.conditions = conditions

    def forward(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()

        # 将条件连接成一个 SQL 语句
        if isinstance(self.conditions, list):
            condition_str = " AND ".join(self.conditions)
        elif isinstance(self.conditions, str):
            condition_str = self.conditions.strip().replace('\n', "").replace(",", " AND ")
        sql_query = f"SELECT * FROM self WHERE {condition_str}"

        self.logger.info(f"[WHERE SQL] {sql_query}")

        # 使用 Polars 的 SQL 方法进行过滤
        filtered_lf = lf.sql(sql_query)

        return filtered_lf.lazy()
