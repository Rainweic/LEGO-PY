import polars as pl
import numpy as np
from stages.split import Split

# 测试脚本
def test_split(split="7:2:1", random=True):
    # 创建测试数据
    test_df = pl.DataFrame({
        "A": range(1000),
        "B": [np.random.randn() for _ in range(1000)]
    })
    
    # 创建Split实例
    splitter = Split(split=split, random=random)
    
    # 执行分割
    train, val, test = splitter.forward(test_df)
    
    # 收集结果
    train_collected = train.collect()
    val_collected = val.collect()
    test_collected = test.collect()

    # print(test_collected)
    
    # 解析分割比例
    ratios = [float(r) for r in split.split(":")]
    total_ratio = sum(ratios)
    expected_sizes = [int(ratio / total_ratio * len(test_df)) for ratio in ratios]
    expected_sizes[-1] = len(test_df) - sum(expected_sizes[:-1])  # 确保总和等于总行数

    print(expected_sizes)
    
    # 检查分割比例是否正确
    assert len(train_collected) == expected_sizes[0], f"训练集大小应为{expected_sizes[0]}，实际为{len(train_collected)}"
    assert len(val_collected) == expected_sizes[1], f"验证集大小应为{expected_sizes[1]}，实际为{len(val_collected)}"
    assert len(test_collected) == expected_sizes[2], f"测试集大小应为{expected_sizes[2]}，实际为{len(test_collected)}"
    
    # 检查是否所有数据都被使用
    total_count = len(train_collected) + len(val_collected) + len(test_collected)
    assert total_count == len(test_df), "总数据量不匹配"
    
    if random:
        # 检查数据是否被打乱
        assert not np.array_equal(train_collected["A"].to_numpy(), np.arange(expected_sizes[0])), "训练集数据未被正确打乱"
        assert not np.array_equal(val_collected["A"].to_numpy(), np.arange(expected_sizes[0], expected_sizes[0] + expected_sizes[1])), "验证集数据未被正确打乱"
        assert not np.array_equal(test_collected["A"].to_numpy(), np.arange(expected_sizes[0] + expected_sizes[1], len(test_df))), f"测试集数据未被正确打乱"
    else:
        # 检查数据是否按顺序分割
        assert np.array_equal(train_collected["A"].to_numpy(), np.arange(expected_sizes[0])), f"训练集数据应该按顺序分割"
        assert np.array_equal(val_collected["A"].to_numpy(), np.arange(expected_sizes[0], expected_sizes[0] + expected_sizes[1])), f"验证集数据应该按顺序分割"
        assert np.array_equal(test_collected["A"].to_numpy(), np.arange(expected_sizes[0] + expected_sizes[1], len(test_df))), "测试集数据应该按顺序分割"
    
    # 检查是否包含所有原始数据
    all_data = pl.concat([train_collected, val_collected, test_collected])
    assert all_data.select(pl.all().sort()).equals(test_df.select(pl.all().sort())), \
        f"分割后的数据与原始数据不匹配，{all_data.select(pl.all().sort())}, {test_df.select(pl.all().sort())}"
    
    print(f"所有测试通过！随机划分: {random}")

# 运行测试
if __name__ == "__main__":
    test_split("7:2:1", random=True)
    test_split("7:2:1", random=False)
