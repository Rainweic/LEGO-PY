import polars as pl
import numpy as np
from stages import Statistics


def test_statistics():
    # 创建测试数据
    test_df = pl.DataFrame({
        "A": [1, 2, 1, None, 5, 10, 15, 20, 25, 30],  # 扩展数值数据便于测试分箱
        "B": [1.1, 2.2, 3.3, 4.4, 5.5, 1.1, 2.2, 3.3, 4.4, 5.5],
        "C": ["a", "b", "c", "a", None, "b", "a", "c", "b", "a"]
    })
    
    # 测试等距分箱
    stats_equal_width = Statistics(cols=["A", "B", "C"], n_bins=5, bin_type='equal_width')
    result_equal_width = stats_equal_width.forward(test_df)
    
    # 测试等频分箱
    stats_equal_freq = Statistics(cols=["A", "B", "C"], n_bins=5, bin_type='equal_freq')
    result_equal_freq = stats_equal_freq.forward(test_df)
    
    # 基本验证
    assert len(result_equal_width) == 3, "应该有3列的统计结果"
    assert len(result_equal_freq) == 3, "应该有3列的统计结果"
    
    # 验证统计结果的列名
    expected_columns = ["特征", "max", "min", "mean", "null_count", "n_unique", "std"]
    assert all(col in result_equal_width.columns for col in expected_columns)
    
    # 验证数值列的统计结果
    a_stats_width = result_equal_width.filter(pl.col("特征") == "A").to_dict(as_series=False)
    a_stats_freq = result_equal_freq.filter(pl.col("特征") == "A").to_dict(as_series=False)
    
    # 验证基本统计量
    assert float(a_stats_width["max"][0]) == 30.0
    assert float(a_stats_width["min"][0]) == 1.0
    assert a_stats_width["null_count"][0] == 1
    
    # 验证分箱结果（通过检查summary中的饼图数据）
    for stats in [stats_equal_width, stats_equal_freq]:
        for summary_item in stats.summary:
            # 验证每个特征都有对应的饼图数据
            assert len(summary_item) == 1
            # 验证饼图选项存在
            assert isinstance(list(summary_item.values())[0], str)
    
    # 测试异常情况
    try:
        invalid_stats = Statistics(cols=["A"], n_bins=5, bin_type='invalid_type')
        invalid_stats.forward(test_df)
        assert False, "应该抛出异常"
    except TypeError as e:
        assert str(e) == "分箱类型仅支持[equal_width, equal_freq]"
    
    print("所有测试通过！")


def test_statistics_edge_cases():
    # 边界情况测试
    test_df = pl.DataFrame({
        "D": [1] * 10,  # 所有值相同
        "E": list(range(10)),  # 连续不重复值
        "F": [None] * 10  # 全是空值
    })
    
    stats = Statistics(cols=["D", "E", "F"], n_bins=5, bin_type='equal_freq')
    result = stats.forward(test_df)
    
    # 验证全相同值的情况
    d_stats = result.filter(pl.col("特征") == "D").to_dict(as_series=False)
    assert float(d_stats["max"][0]) == float(d_stats["min"][0])
    assert d_stats["n_unique"][0] == 1
    
    # 验证全空值的情况
    f_stats = result.filter(pl.col("特征") == "F").to_dict(as_series=False)
    assert f_stats["null_count"][0] == 10
    
    print("边界情况测试通过！")


if __name__ == "__main__":
    test_statistics()
    test_statistics_edge_cases()