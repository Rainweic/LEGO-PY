import polars as pl
import pytest
from stages.nan import DealNan

def test_deal_nan():
    # 创建测试数据
    df = pl.DataFrame({
        'num_col': [1.0, float('nan'), 3.0, 4.0, None],
        'str_col': ['a', '', 'c', None, 'e'],
    }).lazy()
    
    # 测试数值列的各种处理方法
    def test_numeric_methods():
        # 测试均值填充
        stage = DealNan([{'col': 'num_col', 'method': 'mean', 'fill_value': None}])
        result = stage.forward(df).collect()
        print(result)
        assert result['num_col'].mean() == pytest.approx(2.67, rel=1e-2)
        
        # 测试最小值填充
        stage = DealNan([{'col': 'num_col', 'method': 'min', 'fill_value': None}])
        result = stage.forward(df).collect()
        print(result)
        assert result['num_col'].min() == 1.0
        
        # 测试自定义值填充
        stage = DealNan([{'col': 'num_col', 'method': 'custom', 'fill_value': -999}])
        result = stage.forward(df).collect()
        print(result)
        assert -999 in result['num_col'].to_list()
        
        # 测试删除行
        stage = DealNan([{'col': 'num_col', 'method': 'drop', 'fill_value': None}])
        result = stage.forward(df).collect()
        print(result)
        assert len(result) < len(df.collect())

    # 测试字符串列的处理方法
    def test_string_methods():
        # 测试自定义值填充
        stage = DealNan([{'col': 'str_col', 'method': 'custom', 'fill_value': 'MISSING'}])
        result = stage.forward(df).collect()
        assert 'MISSING' in result['str_col'].to_list()
        
        # 测试删除行
        stage = DealNan([{'col': 'str_col', 'method': 'drop', 'fill_value': None}])
        result = stage.forward(df).collect()
        print(result)
        assert len(result) < len(df.collect())

    # 测试多列同时处理
    def test_multiple_columns():
        stage = DealNan([
            {'col': 'num_col', 'method': 'mean', 'fill_value': None},
            {'col': 'str_col', 'method': 'custom', 'fill_value': 'MISSING'}
        ])
        result = stage.forward(df).collect()
        print(result)
        assert not result['num_col'].is_nan().any()
        assert not result['str_col'].is_null().any()

    # 测试空配置
    def test_empty_config():
        stage = DealNan([])
        result = stage.forward(df).collect()
        print(result)
        assert len(result) == len(df.collect())

    # 执行所有测试
    test_numeric_methods()
    test_string_methods()
    test_multiple_columns()
    test_empty_config()

if __name__ == '__main__':
    test_deal_nan() 