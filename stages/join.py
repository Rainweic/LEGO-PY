import pandas as pd
from functools import reduce

from dags.stage import stage


@stage(n_outputs=1)
def join(left_df: pd.DataFrame, right_df: pd.DataFrame, on: list[str], how: str):
    return left_df.join(right_df, on=on, how=how)


@stage(n_outputs=1)
def multi_join(dfs: list[pd.DataFrame], on: list[str], how: str = "left"):
    """
    对多个DataFrame进行连续join操作。

    参数:
    dfs (list[pd.DataFrame]): 要join的DataFrame列表，第一个DataFrame作为基础。
    on (list[str]): 用于join的列名列表。
    how (str): join的方式，默认为'left'。

    返回:
    pd.DataFrame: join后的结果DataFrame。
    """

    def join_two_dfs(left, right):
        return pd.merge(left, right, on=on, how=how)

    return reduce(join_two_dfs, dfs)
