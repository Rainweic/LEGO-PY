import polars as pl
from dags.stage import CustomStage


class BaseNormalize(CustomStage):

    def __init__(self, cols=None):
        super().__init__(n_outputs=1)
        self.cols = cols

    def check_df(self, df: pl.DataFrame):

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        # 检查所有指定的列是否都在DataFrame中
        if self.cols:
            df_columns = set(df.collect_schema().names())
            missing_columns = set(self.cols) - df_columns
            if missing_columns:
                raise ValueError(f"以下列不存在于DataFrame中: {', '.join(missing_columns)}")
            
            # 求df_columns内除了self.cols之外的列
            self.other_columns = df_columns - set(self.cols)
            self.logger.info(f"未被处理的列: {', '.join(self.other_columns)}")
        else:
            self.logger.info("默认对所有特征进行处理")
            self.cols = df.collect_schema().names()
            self.other_columns = []

        return df


class MinMaxNormalize(BaseNormalize):
    """Min-Max Normalization"""

    def forward(self, df: pl.DataFrame):
        
        df = self.check_df(df)

        # 计算每列的最小值和最大值
        min_max = df.select([
            pl.col(col).min().alias(f"min_{col}")
                for col in self.cols
        ] + [
            pl.col(col).max().alias(f"max_{col}")
                for col in self.cols
        ])

        # self.logger.info("min_max DataFrame columns:", min_max.collect_schema().names())

        min_max = min_max.collect()
        
        # 进行归一化
        normalized = df.select([
            ((pl.col(col) - min_max[f"min_{col}"][0]) / (min_max[f"max_{col}"][0] - min_max[f"min_{col}"][0])).alias(col)
            for col in self.cols
        ] + [pl.col(col) for col in self.other_columns])
        
        return normalized
    

class ZScoreNormalize(BaseNormalize):
    """Z Score Normalize"""

    def forward(self, df: pl.DataFrame):
        
        df = self.check_df(df)
        
        # 计算每列的平均值和标准差
        stats = df.select([
            pl.col(col).mean().alias(f"mean_{col}")
                for col in self.cols
        ] + [
            pl.col(col).std().alias(f"std_{col}")
                for col in self.cols
        ])
        
        stats = stats.collect()
        
        # 进行平均归一化
        normalized = df.select([
            ((pl.col(col) - stats[f"mean_{col}"][0]) / stats[f"std_{col}"][0]).alias(col)
            for col in self.cols
        ] + [pl.col(col) for col in self.other_columns])
        
        return normalized


