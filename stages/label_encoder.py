import uuid
import polars as pl
from dags.stage import CustomStage


class CastStage(CustomStage):
    """
    用于对LazyFrame中的列进行值映射的自定义Stage。

    参数:
        col_name (str): 需要进行映射的列名。
        map (dict): 映射字典，键为原始值，值为映射后的值。
        recover_ori_col (bool, 可选): 是否覆盖原始列名。默认为True。
        out_col_name (str, 可选): 如果覆盖原始列名，该参数生效。

    方法:
        forward(df: pl.LazyFrame) -> pl.LazyFrame:
            对输入的LazyFrame进行值映射操作。

    """

    def __init__(self, col_name: str, map: str, recover_ori_col: bool = True, out_col_name: str = None):
        super().__init__(n_outputs=1)
        self.col_name = col_name
        self.map = eval(map)
        self.recover_ori_col = recover_ori_col
        self.out_col_name = out_col_name

    def forward(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        对输入的LazyFrame进行值映射操作。

        参数:
            df (pl.LazyFrame): 输入的LazyFrame。

        返回:
            pl.LazyFrame: 经过值映射操作后的LazyFrame。

        """

        # 将映射字典转换为 SQL CASE WHEN 语句
        case_when_statements = " ".join(
            [f"""WHEN {self.col_name} == {k} THEN {v}""" for k, v in self.map.items()]
        )
        
        if self.recover_ori_col:
            tmp_out_col_name = "casewhen" + str(uuid.uuid4().hex[:8])
            sql_query = f"""
            SELECT 
                *,
                (CASE {case_when_statements} ELSE {self.col_name} END) AS {tmp_out_col_name}
            FROM self
            """.strip()
            self.logger.info(sql_query)
            df = df.lazy().sql(sql_query).drop(pl.col(self.col_name)).rename({tmp_out_col_name: self.col_name})

        else:
            sql_query = f"""
            SELECT 
                *,
                (CASE {case_when_statements} ELSE {self.col_name} END) AS {self.out_col_name}
            FROM self
            """.strip()
            self.logger.info(sql_query)
            df = df.lazy().sql(sql_query)

        return df
