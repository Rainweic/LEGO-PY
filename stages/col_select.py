import polars as pl
from dags.stage import CustomStage


class SelectCols(CustomStage):

    def __init__(self, cols: list[str]):
        super().__init__(n_outputs=1)
        self.cols = cols

    def forward(self, df: pl.DataFrame):
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        return df.select(self.cols)
