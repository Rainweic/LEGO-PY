import polars as pl
from dags.stage import CustomStage


class SqlStage(CustomStage):

    def __init__(self, sql_str: str):
        super().__init__(n_outputs=1)
        self.sql_str = sql_str

    def forward(self, df: pl.DataFrame):
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        df = df.sql(self.sql_str)

        return df
