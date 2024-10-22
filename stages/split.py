import polars as pl
import numpy as np
from dags.stage import CustomStage


class Split(CustomStage):

    def __init__(self, split="7:2:1", random=True):
        super().__init__(n_outputs=3)
        self.split = split
        self.random = random

    def forward(self, df: pl.DataFrame):
        # 确保输入是 LazyFrame
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        # 解析分割比例
        ratios = [float(r) / sum(float(x) for x in self.split.split(":")) for r in self.split.split(":")]
        
        total_rows = df.select(pl.count()).collect().item()
        split_sizes = [int(ratio * total_rows) for ratio in ratios]
        split_sizes[-1] = total_rows - sum(split_sizes[:-1])  # 确保总和等于总行数

        print(f"实际划分样本数量 train:val:test {split_sizes}")
        # self.logger.info(f"实际划分样本数量 train:val:test {split_sizes}")
        
        if self.random:
            # 生成随机索引
            random_indices = np.random.permutation(total_rows)
            
            # 使用随机索引创建一个新的列
            df_with_random = df.with_columns(
                pl.Series(name="__random_index__", values=random_indices)
            )
            
            # 基于随机索引进行分割
            train = df_with_random.filter(pl.col("__random_index__") < split_sizes[0])
            val = df_with_random.filter((pl.col("__random_index__") >= split_sizes[0]) & 
                                        (pl.col("__random_index__") < split_sizes[0] + split_sizes[1]))
            test = df_with_random.filter(pl.col("__random_index__") >= split_sizes[0] + split_sizes[1])
            
            # 移除临时的随机索引列
            train = train.drop("__random_index__")
            val = val.drop("__random_index__")
            test = test.drop("__random_index__")
        else:
            # 顺序划分
            train = df.slice(0, split_sizes[0])
            val = df.slice(split_sizes[0], split_sizes[1])
            test = df.slice(split_sizes[0] + split_sizes[1], split_sizes[2])
        
        return train, val, test
