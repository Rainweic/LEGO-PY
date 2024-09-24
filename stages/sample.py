import polars as pl
import numpy as np

from dags.stage import stage


@stage(n_outputs=1)
def sample_lazyframe(lf: pl.LazyFrame, n: int, seed: int = None) -> pl.LazyFrame:
    """
    从 LazyFrame 中采样 n 行。

    参数:
        lf (pl.LazyFrame): 输入的 LazyFrame。
        n (int): 采样的行数。
        seed (int, optional): 随机种子。

    返回:
        pl.LazyFrame: 采样后的 LazyFrame。
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(lf, pl.DataFrame):
        lf = lf.lazy()
    
    # 获取 LazyFrame 的行数
    num_rows = lf.collect().height
    
    # 生成随机行号
    sampled_indices = np.random.choice(num_rows, n, replace=False)
    
    # 使用 with_row_count 添加行号，然后过滤行号
    lf_sampled = (
        lf.with_row_count("row_nr")
        .filter(pl.col("row_nr").is_in(sampled_indices))
        .drop("row_nr")
    )
    
    return lf_sampled.lazy()
