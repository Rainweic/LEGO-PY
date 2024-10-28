import polars as pl
import numpy as np

from dags.stage import CustomStage


class Sample(CustomStage):

    def __init__(self, n_sample=0, random=False, seed=None):
        super().__init__(n_outputs=1)
        self.n_sample = n_sample
        self.random = random
        self.seed = seed

    def forward(self, lf: pl.LazyFrame):
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        
        # 使用 SQL 来实现随机或顺序采样
        if self.random:
            # 生成随机索引
            total_rows = lf.select(pl.len()).collect().item()
            if self.seed is not None:
                np.random.seed(seed=self.seed)
            random_indices = np.random.permutation(total_rows)
            
            # 使用随机索引创建一个新的列
            lf_with_random = lf.with_columns(
                pl.Series(name="__random_index__", values=random_indices)
            )
            
            # 基于随机索引进行过滤
            result = lf_with_random.filter(pl.col("__random_index__") < self.n_sample)
        
            # 移除临时的随机索引列
            result = result.drop("__random_index__")
            
            return result
        else:
            return lf.slice(0, self.n_sample)
