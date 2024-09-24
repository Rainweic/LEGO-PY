import polars as pl
from tqdm import tqdm
from dags.stage import BaseStage


class MultiJoin(BaseStage):

    def __init__(self, on: str, how: str, col_type: pl.DataType = None):
        super().__init__(n_outputs=1)
        self.on = on
        self.how = how
        self.col_type = col_type

    def forward(self, *dfs: list[pl.LazyFrame]):
        
        def join_two_dfs(left: pl.LazyFrame, right: pl.LazyFrame):
            if isinstance(left, pl.DataFrame):
                left = left.lazy()
            if isinstance(right, pl.DataFrame):
                right = right.lazy()
            if self.col_type:
                left = left.with_columns(pl.col(self.on).cast(self.col_type))
                right = right.with_columns(pl.col(self.on).cast(self.col_type))
            return left.join(right, on=self.on, how=self.how).lazy()
        
        # 使用 tqdm 显示进度条
        result = dfs[0]
        for df in tqdm(dfs[1:], desc="Joining DataFrames"):
            result = join_two_dfs(result, df)

        return result
