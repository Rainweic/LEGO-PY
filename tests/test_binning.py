import pytest
import numpy as np
import pandas as pd
from stages.utils.binning import Binning

@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 创建连续型数据
    values = np.random.normal(35, 10, n_samples)
    values = np.clip(values, 18, 80)  # 限制在合理范围内
    
    # 创建二分类目标变量
    target = np.random.binomial(1, 0.3, n_samples)
    
    return values, target

def test_init():
    """测试初始化"""
    binner = Binning()
    assert binner.method == 'equal_freq'
    assert binner.n_bins == 10
    assert binner.binning_result is None

def test_equal_width_binning(sample_data):
    """测试等宽分箱"""
    values, _ = sample_data
    binner = Binning(method='equal_width', n_bins=5)
    binner.fit(values)
    
    # 检查分箱数量
    assert len(binner.binning_result.bins) == 6  # n_bins + 1
    assert len(binner.binning_result.bin_labels) == 5  # n_bins
    
    # 检查分箱边界是否等间距
    bins = binner.binning_result.bins
    intervals = np.diff(bins)
    assert np.allclose(intervals, intervals[0], rtol=1e-10)

    print(binner.binning_result.bins)
    print(binner.binning_result.bin_labels)
    print(binner.binning_result.bin_stats)

def test_equal_freq_binning(sample_data):
    """测试等频分箱"""
    values, _ = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    binner.fit(values)
    
    # 检查分箱数量
    assert len(binner.binning_result.bins) == 6  # n_bins + 1
    assert len(binner.binning_result.bin_labels) == 5  # n_bins
    
    # 检查每个分箱的样本数量是否接近
    transformed = binner.transform(values)
    bin_counts = pd.Series(transformed).value_counts()
    expected_count = len(values) / 5
    
    # 允许20%的误差范围
    tolerance = expected_count * 0.2
    assert all(abs(count - expected_count) <= tolerance for count in bin_counts)
    
    # 检查分箱边界的单调性
    bins = binner.binning_result.bins
    assert all(bins[i] <= bins[i+1] for i in range(len(bins)-1))
    
    # 检查是否所有值都被分到了箱中
    assert len(transformed) == len(values)
    assert not any(pd.isna(transformed))

def test_chi2_binning(sample_data):
    """测试卡方分箱"""
    values, target = sample_data
    binner = Binning(method='chi2', chi_merge_threshold=0.1)
    binner.fit(values, target)
    
    # 检查分箱结果
    assert len(binner.binning_result.bins) >= 2  # 至少有一个分箱
    assert len(binner.binning_result.bin_labels) == len(binner.binning_result.bins) - 1

def test_kmeans_binning(sample_data):
    """测试KMeans分箱"""
    values, _ = sample_data
    binner = Binning(method='kmeans', n_bins=5)
    binner.fit(values)
    
    # 检查分箱数量
    assert len(binner.binning_result.bins) == 6
    assert len(binner.binning_result.bin_labels) == 5

def test_custom_bins(sample_data):
    """测试自定义分箱"""
    values, _ = sample_data
    custom_bins = [0, 25, 35, 50, float('inf')]
    binner = Binning(custom_bins=custom_bins)
    binner.fit(values)
    
    # 检查是否使用了自定义分箱点
    assert len(binner.binning_result.bins) == len(custom_bins)
    assert all(b1 == b2 for b1, b2 in zip(binner.binning_result.bins, custom_bins))

def test_transform(sample_data):
    """测试转换功能"""
    values, _ = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    
    # 测试fit_transform
    indices = binner.fit_transform(values)
    assert len(indices) == len(values)
    assert all(isinstance(x, (int, np.integer)) for x in indices)
    
    # 测试transform返回标签
    indices, labels = binner.transform(values[:10], return_labels=True)  # 取部分数据测试
    assert len(indices) == len(values[:10])
    assert len(labels) == len(values[:10])
    assert all(isinstance(x, (int, np.integer)) for x in indices)
    assert all(isinstance(x, str) for x in labels)

def test_bin_stats(sample_data):
    """测试分箱统计信息"""
    values, target = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    binner.fit(values, target)
    
    stats = binner.binning_result.bin_stats
    for bin_label, bin_stat in stats.items():
        # 检查必要的统计信息是否存在
        assert 'count' in bin_stat
        assert 'min' in bin_stat
        assert 'max' in bin_stat
        assert 'mean' in bin_stat
        assert 'target_mean' in bin_stat
        assert 'target_count' in bin_stat
        
        # 检查统计值的合理性
        assert bin_stat['count'] > 0
        assert bin_stat['min'] <= bin_stat['max']
        assert 0 <= bin_stat['target_mean'] <= 1

def test_edge_cases():
    """测试边界情况"""
    # 测试空数据
    with pytest.raises(Exception):
        binner = Binning()
        binner.fit(np.array([]))
    
    # 测试无效的分箱方法
    with pytest.raises(ValueError):
        binner = Binning(method='invalid_method')
        binner.fit(np.array([1, 2, 3]))
    
    # 测试在未拟合的情况下调用transform
    with pytest.raises(ValueError):
        binner = Binning()
        binner.transform(np.array([1, 2, 3]))
    
    # 测试缺少目标变量的卡方分箱
    with pytest.raises(ValueError):
        binner = Binning(method='chi2')
        binner.fit(np.array([1, 2, 3]))

def test_mdlp_binning(sample_data):
    """测试MDLP分箱"""
    values, target = sample_data
    binner = Binning(method='mdlp')
    binner.fit(values, target)
    
    # 检查分箱结果
    assert len(binner.binning_result.bins) >= 2  # 至少有一个分箱
    assert len(binner.binning_result.bin_labels) == len(binner.binning_result.bins) - 1

def test_transform_with_indices(sample_data):
    """测试转换功能（返回分箱编号）"""
    values, _ = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    
    # 测试只返回分箱编号
    indices = binner.fit_transform(values)
    assert len(indices) == len(values)
    assert all(isinstance(x, (int, np.integer)) for x in indices)
    assert all(0 <= x < 5 for x in indices)  # 检查编号范围
    
    # 测试同时返回编号和标签
    indices, labels = binner.transform(values, return_labels=True)
    assert len(indices) == len(values)
    assert len(labels) == len(values)
    assert all(isinstance(x, (int, np.integer)) for x in indices)
    assert all(isinstance(x, str) for x in labels)
    assert all(0 <= x < 5 for x in indices)
    
    # 验证编号和标签的对应关系
    for idx, label in zip(indices, labels):
        assert binner.binning_result.bin_indices[idx] == label

def test_bin_indices_mapping(sample_data):
    """测试分箱编号映射"""
    values, _ = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    binner.fit(values)
    
    # 检查bin_indices的完整性
    assert len(binner.binning_result.bin_indices) == 5  # n_bins
    assert all(i in binner.binning_result.bin_indices for i in range(5))
    
    # 检查bin_indices与bin_labels的一致性
    for i, label in binner.binning_result.bin_indices.items():
        assert label == binner.binning_result.bin_labels[i]

def test_transform_edge_values(sample_data):
    """测试边界值的转换"""
    values, _ = sample_data
    binner = Binning(method='equal_freq', n_bins=5)
    binner.fit(values)
    
    # 测试小于最小值的情况
    min_val = values.min()
    indices = binner.transform(np.array([min_val - 1]))
    assert indices[0] == 0  # 应该分到第一个箱
    
    # 测试大于最大值的情况
    max_val = values.max()
    indices = binner.transform(np.array([max_val + 1]))
    assert indices[0] == 4  # 应该分到最后一个箱
    
    # 测试等于边界值的情况
    bins = binner.binning_result.bins
    for i in range(1, len(bins)-1):
        indices = binner.transform(np.array([bins[i]]))
        assert indices[0] == i  # 应该分到右边的箱