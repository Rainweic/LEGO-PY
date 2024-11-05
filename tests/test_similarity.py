import unittest
import polars as pl
import numpy as np
from time import time
from stages.similarity import CustomerSimilarityStage


class TestCustomerSimilarityStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        # 小数据集
        cls.small_group1 = pl.LazyFrame({
            'customer_id': range(1, 11),
            'age_range': np.random.choice(['20-30', '30-40', '40-50'], 10),
            'purchase_category': np.random.choice(['A', 'B', 'C'], 10),
            'location': np.random.choice(['北京', '上海', '广州', '深圳'], 10)
        })
        
        cls.small_group2 = pl.LazyFrame({
            'customer_id': range(11, 21),
            'age_range': np.random.choice(['20-30', '30-40', '40-50'], 10),
            'purchase_category': np.random.choice(['A', 'B', 'C'], 10),
            'location': np.random.choice(['北京', '上海', '广州', '深圳'], 10)
        })
        
        # 大数据集
        np.random.seed(42)
        cls.large_group1 = pl.LazyFrame({
            'customer_id': range(1, 100001),
            'age_range': np.random.choice(['20-30', '30-40', '40-50', '50-60'], 100000),
            'purchase_category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
            'location': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都'], 100000),
            'income_level': np.random.choice(['低', '中', '高'], 100000),
            'education': np.random.choice(['高中', '本科', '硕士', '博士'], 100000)
        })
        
        cls.large_group2 = pl.LazyFrame({
            'customer_id': range(100001, 200001),
            'age_range': np.random.choice(['20-30', '30-40', '40-50', '50-60'], 100000),
            'purchase_category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
            'location': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都'], 100000),
            'income_level': np.random.choice(['低', '中', '高'], 100000),
            'education': np.random.choice(['高中', '本科', '硕士', '博士'], 100000)
        })
        
        # 完全相同的数据集
        cls.identical_group = cls.small_group1
        
        # 完全不同的数据集
        cls.different_group = pl.LazyFrame({
            'customer_id': range(21, 31),
            'age_range': ['60-70'] * 10,
            'purchase_category': ['X'] * 10,
            'location': ['西安'] * 10
        })

    def test_basic_functionality(self):
        """测试基本功能"""
        stage = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location'],
            num_perm=128
        )
        
        result = stage.forward(self.small_group1, self.small_group2)
        
        self.assertIsInstance(result, dict)
        self.assertIn('similarity_score', result)
        self.assertGreaterEqual(result['similarity_score'], 0)
        self.assertLessEqual(result['similarity_score'], 1)

    def test_identical_groups(self):
        """测试完全相同的群体"""
        stage = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location'],
            num_perm=128
        )
        
        result = stage.forward(self.small_group1, self.identical_group)
        self.assertAlmostEqual(result['similarity_score'], 1.0, places=2)

    def test_different_groups(self):
        """测试完全不同的群体"""
        stage = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location'],
            num_perm=128
        )
        
        result = stage.forward(self.small_group1, self.different_group)
        self.assertAlmostEqual(result['similarity_score'], 0.0, places=2)

    def test_sampling_performance(self):
        """测试采样性能"""
        # 不使用采样
        stage_no_sampling = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location', 'income_level', 'education'],
            num_perm=128,
            sample_size=None
        )
        
        # 使用采样
        stage_with_sampling = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location', 'income_level', 'education'],
            num_perm=128,
            sample_size=10000
        )
        
        # 测试性能
        start_time = time()
        result_no_sampling = stage_no_sampling.forward(self.large_group1, self.large_group2)
        time_no_sampling = time() - start_time
        
        start_time = time()
        result_with_sampling = stage_with_sampling.forward(self.large_group1, self.large_group2)
        time_with_sampling = time() - start_time
        
        print(f"\n性能对比:")
        print(f"不使用采样 - 耗时: {time_no_sampling:.2f}秒, 相似度: {result_no_sampling['similarity_score']:.4f}")
        print(f"使用采样 - 耗时: {time_with_sampling:.2f}秒, 相似度: {result_with_sampling['similarity_score']:.4f}")
        
        # 验证采样结果与完整结果的误差在可接受范围内
        self.assertLess(
            abs(result_no_sampling['similarity_score'] - result_with_sampling['similarity_score']),
            0.1  # 允许10%的误差
        )

    def test_parallel_processing(self):
        """测试并行处理性能"""
        # 单线程
        stage_single = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location', 'income_level', 'education'],
            num_perm=128,
            n_threads=1
        )
        
        # 多线程
        stage_multi = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location', 'income_level', 'education'],
            num_perm=128,
            n_threads=4
        )
        
        # 测试性能
        start_time = time()
        result_single = stage_single.forward(self.large_group1, self.large_group2)
        time_single = time() - start_time
        
        start_time = time()
        result_multi = stage_multi.forward(self.large_group1, self.large_group2)
        time_multi = time() - start_time
        
        print(f"\n并行处理性能对比:")
        print(f"单线程 - 耗时: {time_single:.2f}秒, 相似度: {result_single['similarity_score']:.4f}")
        print(f"多线程 - 耗时: {time_multi:.2f}秒, 相似度: {result_multi['similarity_score']:.4f}")
        
        # 验证结果一致性
        self.assertAlmostEqual(
            result_single['similarity_score'],
            result_multi['similarity_score'],
            places=4
        )

    def test_cache_effectiveness(self):
        """测试缓存效果"""
        stage = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location'],
            num_perm=128,
            cache_size=128
        )
        
        # 多次运行相同的数据
        start_time = time()
        first_run = stage.forward(self.small_group1, self.small_group2)
        first_time = time() - start_time
        
        start_time = time()
        second_run = stage.forward(self.small_group1, self.small_group2)
        second_time = time() - start_time
        
        print(f"\n缓存效果对比:")
        print(f"首次运行耗时: {first_time:.4f}秒")
        print(f"二次运行耗时: {second_time:.4f}秒")
        print(f"缓存信息: {second_run.get('performance_info', {}).get('cache_info', {})}")
        
        # 验证缓存命中
        self.assertLess(second_time, first_time)


def run_performance_test():
    """运行性能测试"""
    test = TestCustomerSimilarityStage()
    test.setUpClass()
    
    print("\n=== 开始性能测试 ===")
    
    # 测试不同参数组合
    configs = [
        {"num_perm": 128, "sample_size": None, "n_threads": 1},
        {"num_perm": 128, "sample_size": 10000, "n_threads": 1},
        {"num_perm": 128, "sample_size": 10000, "n_threads": 4},
        {"num_perm": 256, "sample_size": 10000, "n_threads": 4},
    ]
    
    results = []
    for config in configs:
        stage = CustomerSimilarityStage(
            feature_cols=['age_range', 'purchase_category', 'location', 'income_level', 'education'],
            **config
        )
        
        start_time = time()
        result = stage.forward(test.large_group1, test.large_group2)
        elapsed_time = time() - start_time
        
        results.append({
            "config": config,
            "time": elapsed_time,
            "similarity": result['similarity_score']
        })
    
    print("\n性能测试结果:")
    for r in results:
        print(f"\n配置: {r['config']}")
        print(f"耗时: {r['time']:.2f}秒")
        print(f"相似度: {r['similarity']:.4f}")


if __name__ == '__main__':
    # 运行单元测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 运行性能测试
    run_performance_test()