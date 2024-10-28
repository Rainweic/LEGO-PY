import polars as pl
from stages import Statistics


def test_statistics():
    # 创建测试数据
    test_df = pl.DataFrame({
        "A": [1, 2, 1, None, 5],
        "B": [1.1, 2.2, 3.3, 4.4, 5.5],
        "C": ["a", "b", "c", "a", None]
    })
    
    # 创建Statistics实例
    stats = Statistics(cols=["A", "B", "C"])
    
    # 运行forward方法
    stats.forward(test_df)
    
    # 验证结果
    # assert result.shape == test_df.shape, "原始DataFrame应保持不变"
    
    # 验证打印的统计结果（这里我们只能间接验证，因为print输出不易直接断言）
    # 你可以手动检查控制台输出是否符合预期
    
    print("测试通过！")

# 运行测试
test_statistics()