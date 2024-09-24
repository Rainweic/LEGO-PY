import sys
import os
import polars as pl
import numpy as np
from stages.spearman import Spearman
from scipy.stats import spearmanr

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_spearman_with_scipy(df, label_col, cols):
    results = {}
    for col in cols:
        corr, _ = spearmanr(df[col], df[label_col])
        results[f"spearman_corr_{col}"] = corr
    return results

if __name__ == '__main__':
    # 生成更大规模的随机数据
    np.random.seed(42)
    data_size = 100000  # 10万条数据
    df = pl.DataFrame({
        "a": np.random.rand(data_size),
        "b": np.random.rand(data_size),
        "c": np.random.rand(data_size),
        "label": np.random.rand(data_size)
    })

    # 转换为 LazyFrame
    lf = df.lazy()

    # 创建 Spearman 实例，指定列
    spearman_stage = Spearman(label_col="label", cols=["a", "b"])

    # 计算指定列与 label 列的 Spearman 相关系数
    result = spearman_stage.forward(lf)

    # 执行计算并显示结果
    polars_result = result.collect().to_dict(as_series=False)
    print("Polars Spearman Correlation Results:", polars_result)

    # 使用 scipy 计算 Spearman 相关系数进行验证
    scipy_result = calculate_spearman_with_scipy(df.to_pandas(), label_col="label", cols=["a", "b"])
    print("Scipy Spearman Correlation Results:", scipy_result)

    # 断言比较结果
    for key in polars_result:
        assert abs(polars_result[key][0] - scipy_result[key]) < 1e-6, f"Mismatch for {key}: {polars_result[key][0]} vs {scipy_result[key]}"

    # 创建 Spearman 实例，不指定列（默认所有列）
    spearman_stage_all = Spearman(label_col="label")

    # 计算所有列与 label 列的 Spearman 相关系数
    result_all = spearman_stage_all.forward(lf)

    # 执行计算并显示结果
    polars_result_all = result_all.collect().to_dict(as_series=False)
    print("Polars Spearman Correlation Results (All Columns):", polars_result_all)

    # 使用 scipy 计算所有列与 label 列的 Spearman 相关系数进行验证
    scipy_result_all = calculate_spearman_with_scipy(df.to_pandas(), label_col="label", cols=["a", "b", "c"])
    print("Scipy Spearman Correlation Results (All Columns):", scipy_result_all)

    # 断言比较结果
    for key in polars_result_all:
        assert abs(polars_result_all[key][0] - scipy_result_all[key]) < 1e-6, f"Mismatch for {key}: {polars_result_all[key][0]} vs {scipy_result_all[key]}"