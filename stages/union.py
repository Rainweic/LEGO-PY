import polars as pl
import pytest
from dags.stage import CustomStage


class Union(CustomStage):

    def __init__(self, drop_duplicate=True, how='vertical_relaxed', maintain_order=False):
        super().__init__(n_outputs=1)
        self.drop_duplicate = drop_duplicate
        self.how = how
        self.maintain_order = maintain_order

    def forward(self, lf1: pl.LazyFrame, lf2: pl.LazyFrame):
        
        result = pl.concat([lf1, lf2], how=self.how)

        if self.drop_duplicate:
            result = result.select(pl.col("*").unique(maintain_order=True))

        return result

