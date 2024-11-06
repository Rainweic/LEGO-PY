import pytest
import polars as pl
import numpy as np
from stages.woe import WOE

@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 创建年龄数据
    age = np.random.normal(35, 10, n_samples)
    age = np.clip(age, 18, 80)
    
    # 创建收入数据
    income = np.random.exponential(50000, n_samples)
    
    # 创建目标变量 (基于年龄和收入的概率)
    prob = 1 / (1 + np.exp(-(age/50 + income/100000 - 1.5)))
    target = np.random.binomial(1, prob)
    
    df = pl.DataFrame({
        'age': age,
        'income': income,
        'target': target
    })
    
    return df

def test_basic_functionality(sample_data):
    """测试基本功能"""
    woe = WOE(
        cols=['age', 'income'],
        target_col='target',
        binning_method='equal_freq',
        n_bins=5
    )
    
    result = woe.forward(sample_data)
    
    # 检查输出列
    assert 'age_bin' in result.collect_schema().names()
    assert 'age_woe' in result.collect_schema().names()
    assert 'income_bin' in result.collect_schema().names()
    assert 'income_woe' in result.collect_schema().names()
    
    # 检查原始列是否保留
    assert 'age' in result.collect_schema().names()
    assert 'income' in result.collect_schema().names()

def test_custom_bins(sample_data):
    """测试自定义分箱"""
    custom_bins = {
        'age': [0, 25, 35, 50, float('inf')]
    }
    
    woe = WOE(
        cols=['age'],
        target_col='target',
        bins=custom_bins
    )
    
    result = woe.forward(sample_data)
    
    # 检查分箱结果
    unique_bins = result.select('age_bin').unique().collect()
    assert len(unique_bins) <= len(custom_bins['age'])

def test_different_binning_methods(sample_data):
    """测试不同的分箱方法"""
    methods = ['equal_width', 'equal_freq', 'chi2', 'kmeans']
    
    for method in methods:
        woe = WOE(
            cols=['age'],
            target_col='target',
            binning_method=method,
            n_bins=5
        )
        
        result = woe.forward(sample_data)
        assert 'age_bin' in result.collect_schema().names()
        assert 'age_woe' in result.collect_schema().names()

def test_remove_original_columns(sample_data):
    """测试删除原始列"""
    woe = WOE(
        cols=['age', 'income'],
        target_col='target',
        recover_ori_col=False
    )
    
    result = woe.forward(sample_data)
    
    # 检查原始列是否被删除
    assert 'age' not in result.collect_schema().names()
    assert 'income' not in result.collect_schema().names()
    
    # 检查WOE列是否存在
    assert 'age_woe' in result.collect_schema().names()
    assert 'income_woe' in result.collect_schema().names()

def test_single_column_input(sample_data):
    """测试单列输入"""
    woe = WOE(
        cols='age',  # 传入字符串而不是列表
        target_col='target'
    )
    
    result = woe.forward(sample_data)
    assert 'age_bin' in result.collect_schema().names()
    assert 'age_woe' in result.collect_schema().names()

def test_edge_cases(sample_data):
    """测试边界情况"""
    # 测试极少的分箱数
    woe = WOE(
        cols=['age'],
        target_col='target',
        n_bins=2
    )
    result = woe.forward(sample_data)
    assert 'age_woe' in result.collect_schema().names()
    
    # 测试较多的分箱数
    woe = WOE(
        cols=['age'],
        target_col='target',
        n_bins=50
    )
    result = woe.forward(sample_data)
    assert 'age_woe' in result.collect_schema().names()

def test_invalid_inputs():
    """测试无效输入"""
    df = pl.DataFrame({
        'x': [1, 2, 3],
        'target': [0, 1, 1]
    })
    
    # 测试无效的分箱方法
    with pytest.raises(ValueError):
        woe = WOE(
            cols=['x'],
            target_col='target',
            binning_method='invalid_method'
        )
        woe.forward(df)
    
    # 测试不存在的列
    with pytest.raises(Exception):
        woe = WOE(
            cols=['non_existent_col'],
            target_col='target'
        )
        woe.forward(df)

def test_woe_values(sample_data):
    """测试WOE值的合理性"""
    woe = WOE(
        cols=['age'],
        target_col='target',
        n_bins=5
    )
    
    result = woe.forward(sample_data).collect()
    woe_values = result['age_woe'].drop_nulls().unique()
    
    # WOE值应该是有限的浮点数
    assert all(np.isfinite(v) for v in woe_values)
    
    # 检查WOE值的范围是否合理
    assert all(abs(v) < 10 for v in woe_values)  # WOE值通常不会太大

def test_lazy_evaluation(sample_data):
    """测试惰性求值"""
    lazy_df = sample_data.lazy()
    
    woe = WOE(
        cols=['age'],
        target_col='target'
    )
    
    result = woe.forward(lazy_df)
    assert isinstance(result, pl.LazyFrame)
    
    # 确保可以正确收集结果
    collected = result.collect()
    assert isinstance(collected, pl.DataFrame)