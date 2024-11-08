import unittest
import polars as pl
import numpy as np
from time import time
from lshashpy3.lshash import LSHash

from sklearn.calibration import LabelEncoder
from stages.similarity import CustomerSimilarityStage


class TestCustomerSimilarityStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """准备真实测试数据"""
        # 设置特征列
        cls.feature_cols = ['age_range', 'city', 'category', 'spending_level', 'member_level']
        
        # 生成用户行为数据
        np.random.seed(42)
        n_users = 50000
        
        # 更真实的用户属性分布
        age_ranges = np.random.choice(
            ['18-24', '25-34', '35-44', '45-54', '55+'],
            n_users,
            p=[0.2, 0.35, 0.25, 0.15, 0.05]  # 年龄分布偏向年轻群体
        )
        
        cities = np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆'],
            n_users,
            p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]  # 一线和新一线城市分布
        )
        
        # 购物类别 - 基于常见电商类别
        categories = np.random.choice(
            ['服装', '数码', '美妆', '食品', '家居', '母婴', '运动', '图书'],
            n_users,
            p=[0.25, 0.15, 0.15, 0.12, 0.12, 0.08, 0.08, 0.05]
        )
        
        # 消费水平
        spending_levels = np.random.choice(
            ['低', '中低', '中', '中高', '高'],
            n_users,
            p=[0.1, 0.2, 0.4, 0.2, 0.1]  # 正态分布
        )
        
        # 会员等级
        member_levels = np.random.choice(
            ['普通会员', '银卡会员', '金卡会员', '钻石会员'],
            n_users,
            p=[0.4, 0.3, 0.2, 0.1]  # 金字塔分布
        )
        
        # 创建两个不同的用户群体
        # 群体1：年轻、高消费群体
        young_high_spenders_mask = (
            (age_ranges == '25-34') & 
            (np.isin(spending_levels, ['中高', '高']))
        )
        
        # 群体2：中年、中等消费群体
        middle_age_mid_spenders_mask = (
            (age_ranges == '35-44') & 
            (np.isin(spending_levels, ['中']))
        )
        
        # 创建数据框
        cls.group1_data = pl.LazyFrame({
            'customer_id': range(1, sum(young_high_spenders_mask) + 1),
            'age_range': age_ranges[young_high_spenders_mask],
            'city': cities[young_high_spenders_mask],
            'category': categories[young_high_spenders_mask],
            'spending_level': spending_levels[young_high_spenders_mask],
            'member_level': member_levels[young_high_spenders_mask]
        })
        
        cls.group2_data = pl.LazyFrame({
            'customer_id': range(1, sum(middle_age_mid_spenders_mask) + 1),
            'age_range': age_ranges[middle_age_mid_spenders_mask],
            'city': cities[middle_age_mid_spenders_mask],
            'category': categories[middle_age_mid_spenders_mask],
            'spending_level': spending_levels[middle_age_mid_spenders_mask],
            'member_level': member_levels[middle_age_mid_spenders_mask]
        })
        
        # 添加边界测试数据
        # 1. 空数据集
        cls.empty_data = pl.LazyFrame({
            'customer_id': [],
            'age_range': [],
            'city': [],
            'category': [],
            'spending_level': [],
            'member_level': []
        })
        
        # 2. 单条数据
        cls.single_record_data = pl.LazyFrame({
            'customer_id': [1],
            'age_range': ['25-34'],
            'city': ['北京'],
            'category': ['数码'],
            'spending_level': ['高'],
            'member_level': ['钻石会员']
        })
        
        # 3. 含空值的数据集
        cls.data_with_nulls = pl.LazyFrame({
            'customer_id': range(1, 11),
            'age_range': ['25-34'] * 5 + [None] * 5,
            'city': ['北京', None] * 5,
            'category': [None] * 5 + ['数码'] * 5,
            'spending_level': ['高'] * 8 + [None] * 2,
            'member_level': ['钻石会员'] * 7 + [None] * 3
        })
        
        # 4. 极端值数据集
        cls.extreme_data = pl.LazyFrame({
            'customer_id': range(1, 11),
            'age_range': ['100+'] * 10,
            'city': ['其他'] * 10,
            'category': ['未知类别'] * 10,
            'spending_level': ['特殊'] * 10,
            'member_level': ['测试会员'] * 10
        })

    def _prepare_vector_data(self, df: pl.LazyFrame) -> np.ndarray:
        """将数据转换为向量形式"""
        df_collected = df.collect()
        
        # 为每个特征列创建独立的编码器
        encoders = {}
        vector_data = []
        
        for col in self.feature_cols:
            encoders[col] = LabelEncoder()
            # 确保处理空值
            col_values = df_collected.get_column(col).fill_null("MISSING").to_numpy()
            encoded_values = encoders[col].fit_transform(col_values)
            vector_data.append(encoded_values)
        
        # 转换为特征矩阵
        return np.column_stack(vector_data)

    # def test_basic_functionality(self):
    #     """基本功能测试"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
    #     result = stage.forward(self.group1_data, self.group2_data)
        
    #     self.assertGreaterEqual(result['similarity_score'], 0.0)
    #     self.assertLessEqual(result['similarity_score'], 1.0)
    #     self.assertIn('details', result)

    # def test_identical_groups(self):
    #     """测试完全相同的群体"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     result = stage.forward(self.group1_data, self.group1_data)
    #     self.assertAlmostEqual(result['similarity_score'], 1.0, places=2)

    # def test_different_groups(self):
    #     """测试完全不同的群体"""
    #     print("测试完全不同的群体")
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     # 创建两个完全不同的群体
    #     group1 = pl.LazyFrame({
    #         'customer_id': [1, 2, 3],
    #         'age_range': ['18-24'] * 3,
    #         'city': ['北京'] * 3,
    #         'category': ['数码'] * 3,
    #         'spending_level': ['高'] * 3,
    #         'member_level': ['钻石会员'] * 3
    #     })
        
    #     group2 = pl.LazyFrame({
    #         'customer_id': [4, 5, 6],
    #         'age_range': ['55+'] * 3,
    #         'city': ['上海'] * 3,
    #         'category': ['服装'] * 3,
    #         'spending_level': ['低'] * 3,
    #         'member_level': ['普通会员'] * 3
    #     })
        
    #     result = stage.forward(group1, group2)
        
    #     # 验证结果
    #     self.assertLess(result['similarity_score'], 0.1)
    #     self.assertEqual(result['details']['intersection_size'], 0)
    #     self.assertGreater(result['details']['union_size'], 0)

    # def test_empty_data(self):
    #     """测试空数据集"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     # 测试两个空数据集
    #     result1 = stage.forward(self.empty_data, self.empty_data)
    #     self.assertEqual(result1['similarity_score'], 0.0)
        
    #     # 测试一个空一个非空
    #     result2 = stage.forward(self.empty_data, self.group1_data)
    #     self.assertEqual(result2['similarity_score'], 0.0)

    # def test_single_record(self):
    #     """测试单条记录"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     # 测试单条记录与自身
    #     result1 = stage.forward(self.single_record_data, self.single_record_data)
    #     self.assertEqual(result1['similarity_score'], 1.0)
        
    #     # 测试单条记录与多条记录
    #     result2 = stage.forward(self.single_record_data, self.group1_data)
    #     self.assertGreaterEqual(result2['similarity_score'], 0.0)
    #     self.assertLessEqual(result2['similarity_score'], 1.0)

    # def test_null_values(self):
    #     """测试空值处理"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     # 测试含空值的数据集
    #     result = stage.forward(self.data_with_nulls, self.group1_data)
    #     self.assertGreaterEqual(result['similarity_score'], 0.0)
    #     self.assertLessEqual(result['similarity_score'], 1.0)
        
    #     # 验证空值不会导致错误
    #     self.assertIn('details', result)
    #     self.assertIn('performance_info', result)

    # def test_extreme_values(self):
    #     """测试极端值"""
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #        )
        
    #     # 测试极端值数据集
    #     result = stage.forward(self.extreme_data, self.group1_data)
    #     self.assertAlmostEqual(result['similarity_score'], 0.0, places=2)
        
    #     # 验证极端值数据集与自身的相似度
    #     result_self = stage.forward(self.extreme_data, self.extreme_data)
    #     self.assertEqual(result_self['similarity_score'], 1.0)

    # def test_feature_subset(self):
    #     """测试特征子集"""
    #     stage_subset = CustomerSimilarityStage(
    #         feature_cols=['age_range', 'spending_level'],
    #     )
        
    #     stage_full = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #     )
        
    #     result_subset = stage_subset.forward(self.group1_data, self.group2_data)
    #     result_full = stage_full.forward(self.group1_data, self.group2_data)
        
    #     # 验证结果不同但都在有效范围内
    #     self.assertNotEqual(result_subset['similarity_score'], result_full['similarity_score'])
    #     self.assertGreaterEqual(result_subset['similarity_score'], 0.0)
    #     self.assertLessEqual(result_subset['similarity_score'], 1.0)

    def test_weights_configuration(self):
        """测试权重配置功能"""
        # 测试自动权重
        stage_auto = CustomerSimilarityStage(
            feature_cols=self.feature_cols,
            weights="auto"
        )
        result_auto = stage_auto.forward(self.group1_data, self.group2_data)
        
        # 测试极端权重差异
        stage_manual = CustomerSimilarityStage(
            feature_cols=self.feature_cols,
            weights={
                'age_range': 10,      # 年龄极其重要
                'spending_level': 8,   # 消费水平很重要
                'member_level': 1,     # 其他特征权重较小
                'city': 1,
                'category': 1
            }
        )
        result_manual = stage_manual.forward(self.group1_data, self.group2_data)
        
        # 测试相反的权重配置
        stage_reverse = CustomerSimilarityStage(
            feature_cols=self.feature_cols,
            weights={
                'age_range': 1,       # 年龄不重要
                'spending_level': 1,   # 消费水平不重要
                'member_level': 10,    # 会员等级极其重要
                'city': 8,            # 城市很重要
                'category': 8         # 类别很重要
            }
        )
        result_reverse = stage_reverse.forward(self.group1_data, self.group2_data)
        
        # 验证结果
        self.assertNotEqual(result_auto['similarity_score'], result_manual['similarity_score'])
        self.assertNotEqual(result_manual['similarity_score'], result_reverse['similarity_score'])
        
        # 验证权重效果的合理性
        # 由于测试数据中年龄和消费水平差异较大，增加这些特征的权重应该降低相似度
        self.assertLess(result_manual['similarity_score'], result_reverse['similarity_score'])


def run_performance_test():
    """运行性能测试，包括与LSH的对比"""
    test = TestCustomerSimilarityStage()
    test.setUpClass()
    
    print("\n=== 开始性能测试 ===")
    
    configs = [
        {"num_perm": 64, "threshold": 0.1},
        {"num_perm": 128, "threshold": 0.01},
        {"num_perm": 256, "threshold": 0.001},
    ]
    
    for config in configs:
        print(f"\n配置: {config}")
        
        # 我们的算法
        stage = CustomerSimilarityStage(
            feature_cols=test.feature_cols,
            weights="auto",  # 使用自动权重
            **config
        )
        
        start_time = time()
        our_result = stage.forward(test.group1_data, test.group2_data)
        our_time = time() - start_time
        
        # LSH算法
        vector_data1 = test._prepare_vector_data(test.group1_data)
        vector_data2 = test._prepare_vector_data(test.group2_data)
        
        # 标准化向量
        vector_data1_norm = vector_data1 / np.linalg.norm(vector_data1, axis=1)[:, np.newaxis]
        vector_data2_norm = vector_data2 / np.linalg.norm(vector_data2, axis=1)[:, np.newaxis]
        
        start_time = time()
        lsh = LSHash(hash_size=config['hash_size'], input_dim=vector_data1.shape[1], num_hashtables=config['num_hashtables'])
        
        # 跳过包含NaN的向量
        valid_vecs1 = vector_data1_norm[~np.any(np.isnan(vector_data1_norm), axis=1)]
        valid_vecs2 = vector_data2_norm[~np.any(np.isnan(vector_data2_norm), axis=1)]
        
        for vec in valid_vecs1:
            lsh.index(vec)
            
        similarities = []
        for vec in valid_vecs2:
            nearest = lsh.query(vec, num_results=1, distance_func="cosine")
            if nearest:
                similarities.append(1 - nearest[0][1])
        
        lsh_time = time() - start_time
        lsh_similarity = np.mean(similarities) if similarities else 0.0
        
        print(f"我们的算法 - 耗时: {our_time:.2f}秒, 相似度: {our_result['similarity_score']:.4f}")
        print(f"LSH算法 - 耗时: {lsh_time:.2f}秒, 相似度: {lsh_similarity:.4f}")


if __name__ == '__main__':
    # 运行单元测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 运行性能测试
    # run_performance_test()