import polars as pl
import pytest
from stages import LabelEncoder


def test_label_encoder_basic():
    # 创建测试数据，使用字符串类型
    test_df = pl.DataFrame({
        "category": ["A", "B", "C", "A", "B", "C", "A"],
        "status": ["active", "inactive", "active", "inactive", "active", "active", "inactive"],
        "number": [1, 2, 3, 1, 2, 3, 1]  # 数值类型，应该被跳过
    })
    
    # 测试单列编码，不覆盖原始列
    encoder_single = LabelEncoder(cols=["category"], replace_original=False)
    result_single = encoder_single.forward(test_df).collect()
    print(result_single)
    
    # 验证结果
    assert "category" in result_single.columns  # 原始列应该保留
    assert "category_encoded" in result_single.columns
    assert set(result_single["category_encoded"].unique()) == {0, 1, 2}
    assert len(result_single) == len(test_df)
    
    # 测试多列编码，覆盖原始列
    encoder_multi = LabelEncoder(cols=["category", "status"], replace_original=True)
    result_multi = encoder_multi.forward(test_df).collect()
    print(result_multi)
    
    # 验证结果
    assert "category" in result_multi.columns  # 原始列被编码值替换
    assert "status" in result_multi.columns
    assert "category_encoded" not in result_multi.columns  # 不应该有额外的编码列
    assert "status_encoded" not in result_multi.columns
    
    print("基本功能测试通过！")


def test_label_encoder_edge_cases():
    # 测试边界情况，使用字符串类型
    edge_df = pl.DataFrame({
        "single_value": ["A"] * 5,  # 只有一个唯一值
        "all_unique": ["A", "B", "C", "D", "E"],  # 所有值都不同
        "with_null": ["A", None, "B", None, "C"],  # 包含空值
        "empty_strings": ["", "", "", "", ""],  # 空字符串
    })
    
    # 测试不覆盖原始列
    encoder = LabelEncoder(cols=["single_value", "all_unique", "with_null", "empty_strings"], 
                         replace_original=False)
    result = encoder.forward(edge_df).collect()
    print(result)
    
    # 验证单一值的编码
    assert len(result["single_value_encoded"].unique()) == 1
    
    # 验证全唯一值的编码
    assert len(result["all_unique_encoded"].unique()) == 5
    
    # 验证空值处理
    null_encoded = result.filter(pl.col("with_null").is_null())["with_null_encoded"]
    assert all(val == -1 for val in null_encoded)
    
    # 验证空字符串处理
    assert len(result["empty_strings_encoded"].unique()) == 1
    
    print("边界情况测试通过！")


def test_label_encoder_type_handling():
    # 测试不同数据类型的处理
    mixed_df = pl.DataFrame({
        "strings": ["A", "B", "C"],  # 应该被编码
        "integers": [1, 2, 3],  # 应该被跳过
        "floats": [1.1, 2.2, 3.3],  # 应该被跳过
        "booleans": [True, False, True]  # 应该被跳过
    })
    
    # 测试覆盖原始列
    encoder = LabelEncoder(cols=["strings", "integers", "floats", "booleans"], 
                         replace_original=True)
    result = encoder.forward(mixed_df).collect()
    print(result)
    
    # 验证只有字符串类型的列被编码
    assert isinstance(result["strings"][0], int)  # 字符串列应该被编码为整数
    assert isinstance(result["integers"][0], int)  # 数值列应该保持原样
    assert isinstance(result["floats"][0], float)
    assert isinstance(result["booleans"][0], bool)
    
    print("数据类型处理测试通过！")


if __name__ == "__main__":
    test_label_encoder_basic()
    test_label_encoder_edge_cases()
    test_label_encoder_type_handling()
    print("所有测试通过！")