import polars as pl
from pyecharts.components import Table
from pyecharts import options as opts
from dags.stage import CustomStage
from utils.convert import chart_2_html


class Statistics(CustomStage):

    def __init__(self, cols=[]):
        super().__init__(n_outputs=1)
        self.cols = cols

    def forward(self, df: pl.LazyFrame):
        stats_list = []
        for col in self.cols:
            col_dtype = str(df.schema[col])
            # print(col_dtype)
            col_stats = df.lazy().select([
                pl.lit(col).alias("特征"),
                pl.col(col).max().cast(pl.Utf8).alias("max"),
                pl.col(col).min().cast(pl.Utf8).alias("min"),
                pl.col(col).mean().cast(pl.Float64).alias("mean"),
                pl.col(col).null_count().cast(pl.Int64).alias("null_count"),
                pl.col(col).n_unique().cast(pl.Int64).alias("n_unique"),
                pl.col(col).std().cast(pl.Float64).alias("std") if col_dtype != "String" else pl.lit(None).cast(pl.Float64).alias("标准差"),
            ])
            stats_list.append(col_stats)
        stats: pl.DataFrame = pl.concat(stats_list).collect()

        return stats
