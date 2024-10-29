import pytest
import polars as pl
from stages.union import Union


if __name__ == "__main__":
    # 创建测试数据
    df1 = pl.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    }).lazy()
    
    df2 = pl.DataFrame({
        "a": [3, 4, 5], 
        "b": ["z", "w", "v"]
    }).lazy()
    
    # 测试带重复的合并
    union = Union(drop_duplicate=False)
    result = union.forward(df1, df2).collect()
    print("带重复的合并结果:")
    print(result)
    assert len(result) == 6
    assert result["a"].to_list() == [1, 2, 3, 3, 4, 5]
    assert result["b"].to_list() == ["x", "y", "z", "z", "w", "v"]
    
    # 测试去重的合并
    union = Union(drop_duplicate=True)
    result = union.forward(df1, df2).collect()
    print("\n去重后的合并结果:")
    print(result)
    assert len(result) == 5
    assert result["a"].to_list() == [1, 2, 3, 4, 5]
    assert result["b"].to_list() == ["x", "y", "z", "w", "v"]