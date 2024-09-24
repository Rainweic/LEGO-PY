import sys
import os
import polars as pl
import numpy as np
from stages.pearson import Pearson
from scipy.stats import pearsonr

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_pearson_with_scipy(df, label_col, cols):
    results = {}
    for col in cols:
        corr, _ = pearsonr(df[col], df[label_col])
        results[f"pearson_corr_{col}"] = corr
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

    # 创建 Pearson 实例，指定列
    pearson_stage = Pearson(label_col="label", cols=["a", "b"])

    # 计算指定列与 label 列的 Pearson 相关系数
    result = pearson_stage.forward(lf)

    # 执行计算并显示结果
    polars_result = result.collect().to_dict(as_series=False)
    print("Polars Pearson Correlation Results:", polars_result)

    # 使用 scipy 计算 Pearson 相关系数进行验证
    scipy_result = calculate_pearson_with_scipy(df.to_pandas(), label_col="label", cols=["a", "b"])
    print("Scipy Pearson Correlation Results:", scipy_result)

    # 断言比较结果
    for key in polars_result:
        assert abs(polars_result[key][0] - scipy_result[key]) < 1e-6, f"Mismatch for {key}: {polars_result[key][0]} vs {scipy_result[key]}"

    # 创建 Pearson 实例，不指定列（默认所有列）
    pearson_stage_all = Pearson(label_col="label")

    # 计算所有列与 label 列的 Pearson 相关系数
    result_all = pearson_stage_all.forward(lf)

    # 执行计算并显示结果
    polars_result_all = result_all.collect().to_dict(as_series=False)
    print("Polars Pearson Correlation Results (All Columns):", polars_result_all)

    # 使用 scipy 计算所有列与 label 列的 Pearson 相关系数进行验证
    scipy_result_all = calculate_pearson_with_scipy(df.to_pandas(), label_col="label", cols=["a", "b", "c"])
    print("Scipy Pearson Correlation Results (All Columns):", scipy_result_all)

    # 断言比较结果
    for key in polars_result_all:
        assert abs(polars_result_all[key][0] - scipy_result_all[key]) < 1e-6, f"Mismatch for {key}: {polars_result_all[key][0]} vs {scipy_result_all[key]}"