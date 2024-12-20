import polars as pl
from dags.stage import BaseStage

class Spearman(BaseStage):

    def __init__(self, label_col: str, cols: list[str] = None, exclude_cols: list[str] = None):
        super().__init__(n_outputs=1)
        self.label_col = label_col
        self.cols = cols
        self.exclude_cols = exclude_cols + [self.label_col]

    def forward(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()

        # 如果没有指定列，则默认选择所有列，但排除不参与计算的列
        if self.cols is None:
            self.logger.warning(f"参与计算的列为空，默认选择剔除{self.exclude_cols}之外的所有列进行计算")
            self.cols = [col for col in lf.columns if col not in self.exclude_cols]
        else:
            self.cols = [col for col in self.cols if col not in self.exclude_cols]
        self.logger.warning(f"参与计算的列为{self.cols}")

        # 将列转换为秩
        rank_lf = lf.with_columns([
            pl.col(col).rank().alias(col) for col in self.cols + [self.label_col]
        ])

        # 计算每个指定列与 label 列的 Spearman 相关系数
        spearman_corrs = []
        for col in self.cols:
            spearman_corrs.append(
                (pl.col(col) * pl.col(self.label_col)).mean() - (pl.col(col).mean() * pl.col(self.label_col).mean())
            )
            spearman_corrs[-1] = spearman_corrs[-1] / (pl.col(col).std() * pl.col(self.label_col).std())
            spearman_corrs[-1] = spearman_corrs[-1]

        result = rank_lf.select(spearman_corrs)

        return result