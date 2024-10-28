import polars as pl
from stages import MinMaxNormalize, ZScoreNormalize


def test_min_max():
    # 创建测试数据
    test_data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    df = pl.DataFrame(test_data)

    # 实例化Normalize类
    normalizer = MinMaxNormalize()

    # 调用forward方法
    result = normalizer.forward(df).collect()

    print("归一化结果:")
    print(result)

    # 使用assert进行正确性判断
    for col in result.columns:
        min_val = result[col].min()
        max_val = result[col].max()
        
        print(f"{col} 列的最小值: {min_val}, 最大值: {max_val}")
        
        assert abs(min_val) < 1e-6, f"{col} 列的最小值应该接近0"
        assert abs(max_val - 1) < 1e-6, f"{col} 列的最大值应该接近1"
        
        assert all(0 <= val <= 1 for val in result[col]), f"{col} 列的所有值应该在0到1之间"

    print("所有测试通过！")


def test_avg_normalize():
    # 创建测试数据
    test_data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    df = pl.DataFrame(test_data)

    # 实例化AvgNormalize类
    normalizer = ZScoreNormalize()

    # 调用forward方法
    result = normalizer.forward(df).collect()

    print("平均归一化结果:")
    print(result)

    # 使用assert进行正确性判断
    for col in result.columns:
        mean_val = result[col].mean()
        std_val = result[col].std()
        
        print(f"{col} 列的平均值: {mean_val}, 标准差: {std_val}")
        
        assert abs(mean_val) < 1e-6, f"{col} 列的平均值应该接近0"
        assert abs(std_val - 1) < 1e-6, f"{col} 列的标准差应该接近1"

    print("所有测试通过！")


if __name__ == "__main__":
    test_min_max()
    test_avg_normalize()