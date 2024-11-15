import polars as pl
import pytest
import os
from stages.histogram import Histogram

class TestHistogram:
    @pytest.fixture
    def sample_data(self):
        # 创建测试数据
        return pl.DataFrame({
            # 数值列
            'age': [25, 30, 35, 40, 45, 25, 30, 35, 40, 45],
            'salary': [5000, 6000, 7000, 8000, 9000, 5500, 6500, 7500, 8500, 9500],
            # 类别列
            'city': ['北京', '上海', '广州', '深圳', '北京', 
                    '上海', '北京', '上海', '广州', '深圳'],
            'education': ['本科', '硕士', '博士', '本科', '硕士',
                         '本科', '本科', '硕士', '本科', '硕士']
        }).lazy()

    @pytest.fixture
    def custom_bins(self):
        return {
            'age': [20, 30, 40, 50],
            'salary': [4000, 6000, 8000, 10000]
        }

    def test_histogram_initialization(self):
        # 测试初始化
        hist = Histogram(method='equal_width', n_bins=5)
        assert hist.method == 'equal_width'
        assert hist.n_bins == 5
        assert hist.custom_bins is None

        # 测试自定义bins的初始化
        hist_custom = Histogram(custom_bins='{"age": [20, 30, 40, 50]}')
        assert isinstance(hist_custom.custom_bins, dict)
        assert 'age' in hist_custom.custom_bins

    def test_numeric_column_processing(self, sample_data):
        hist = Histogram(n_bins=5)
        results = hist.forward(sample_data)
        
        # 检查是否生成了所有列的直方图
        assert 'age' in results
        assert 'salary' in results
        
        # 检查文件是否生成
        assert os.path.exists(results['age'])
        assert os.path.exists(results['salary'])
        
        # 清理生成的文件
        for file in results.values():
            if os.path.exists(file):
                os.remove(file)

    def test_categorical_column_processing(self, sample_data):
        hist = Histogram(n_bins=3)  # 设置为3个箱，应该会有2个单独的类别和1个Others
        results = hist.forward(sample_data)
        
        # 检查类别列的直方图
        assert 'city' in results
        assert 'education' in results
        
        # 检查文件是否生成
        assert os.path.exists(results['city'])
        assert os.path.exists(results['education'])
        
        # 清理生成的文件
        for file in results.values():
            if os.path.exists(file):
                os.remove(file)

    def test_custom_bins(self, sample_data, custom_bins):
        hist = Histogram(custom_bins=custom_bins)
        results = hist.forward(sample_data)
        
        # 检查使用自定义bins的列
        assert 'age' in results
        assert 'salary' in results
        
        # 清理生成的文件
        for file in results.values():
            if os.path.exists(file):
                os.remove(file)

    def test_edge_cases(self):
        # 测试空数据框
        empty_df = pl.DataFrame({
            'age': [],
            'city': []
        }).lazy()
        
        hist = Histogram()
        with pytest.raises(Exception):  # 应该抛出异常
            hist.forward(empty_df)

        # 测试全部为空值的列
        null_df = pl.DataFrame({
            'age': [None, None, None],
            'city': [None, None, None]
        }).lazy()
        
        with pytest.raises(Exception):  # 应该抛出异常
            hist.forward(null_df)

    def test_different_bin_sizes(self, sample_data):
        # 测试不同的bin数量
        for n_bins in [3, 5, 10]:
            hist = Histogram(n_bins=n_bins)
            results = hist.forward(sample_data)
            
            # 检查是否所有文件都生成了
            assert len(results) == len(sample_data.collect().columns)
            
            # 清理文件
            for file in results.values():
                if os.path.exists(file):
                    os.remove(file)

    @pytest.mark.parametrize("method", ['equal_width', 'equal_freq'])
    def test_different_methods(self, sample_data, method):
        # 测试不同的分箱方法
        hist = Histogram(method=method)
        results = hist.forward(sample_data)
        
        # 检查是否所有文件都生成了
        assert len(results) == len(sample_data.collect().columns)
        
        # 清理文件
        for file in results.values():
            if os.path.exists(file):
                os.remove(file)