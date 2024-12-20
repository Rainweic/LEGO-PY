import unittest
import polars as pl
import numpy as np
from time import time
from lshashpy3.lshash import LSHash

from sklearn.calibration import LabelEncoder
from stages.similarity import CustomerSimilarityStage
from stages.similarity import BinnedKLSimilarityStage


class TestCustomerSimilarityStage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """准备真实测试数据"""
        # 设置特征列
        cls.feature_cols = ['age_range', 'city', 'category', 'spending_level', 'member_level']
        
        # 生成用户行为数据
        np.random.seed(42)
        n_users = 100000
        
        # 更真实的用户属性分布
        age_ranges = np.random.choice(
            ['18-24', '25-34', '35-44', '45-54', '55+'],
            n_users,
            p=[0.2, 0.35, 0.25, 0.15, 0.05]  # 年龄分布偏向年轻群体
        )
        
        cities = np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆'],
            n_users,
            p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]  # 线和新一线城市分布
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
        
    #     # 创建两个完全不���的群体
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
    #     self.assertLess(result['similarity_score'], 0.3)
    #     # self.assertEqual(result['details']['intersection_size'], 0)
    #     # self.assertGreater(result['details']['union_size'], 0)

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

    # def test_similar_groups(self):
    #     """测试高度相似的群体"""
    #     print("测试高度相似的群体")
    #     stage = CustomerSimilarityStage(
    #         feature_cols=self.feature_cols,
    #     )
        
    #     # 创建两个高度相似的群体
    #     # 只有少量差异的属性
    #     group1 = pl.LazyFrame({
    #         'customer_id': range(1, 11),
    #         'age_range': ['25-34'] * 10,
    #         'city': ['北京'] * 5 + ['上海'] * 5,
    #         'category': ['数码'] * 7 + ['电子'] * 3,  # 相似类别
    #         'spending_level': ['高'] * 8 + ['中高'] * 2,  # 相近消费水平
    #         'member_level': ['钻石会员'] * 6 + ['金卡会员'] * 4  # 相近会员等级
    #     })
        
    #     group2 = pl.LazyFrame({
    #         'customer_id': range(11, 21),
    #         'age_range': ['25-34'] * 8 + ['26-35'] * 2,  # 略微年龄差异
    #         'city': ['北京'] * 6 + ['海'] * 4,  # 相似城市分布
    #         'category': ['数码'] * 8 + ['电子'] * 2,
    #         'spending_level': ['高'] * 7 + ['中高'] * 3,
    #         'member_level': ['钻石会员'] * 5 + ['金卡会员'] * 5
    #     })
        
    #     result = stage.forward(group1, group2)
        
    #     # 验证结果
    #     print(f"相似群体的相似度得分: {result['similarity_score']:.3f}")
        
    #     # 相似度应该很高，但不完全相同
    #     self.assertGreater(result['similarity_score'], 0.7)  # 相似度应该大于0.7
    #     self.assertLess(result['similarity_score'], 1.0)     # 但不应该完全相同
        
    #     # 验证详细信息
    #     self.assertIn('details', result)
    #     self.assertIn('cosine_similarity', result['details'])
    #     self.assertIn('euclidean_distance', result['details'])
        
    #     # 余弦相似度应该也很高
    #     self.assertGreater(result['details']['cosine_similarity'], 0.7)


class TestBinnedKLSimilarityStage(unittest.TestCase):
    """测试基于分箱和KL散度的相似度计算"""
    
    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        # 设置特征列
        cls.feature_cols = ['age_range', 'city', 'category', 'spending_level', 'member_level']
        
        # 生成基础测试数据
        np.random.seed(42)
        n_users = 100000
        
        # 1. 生成两个相似但不完全相同的用户群
        # 群体1: 年轻用户群
        young_age_ranges = np.random.choice(
            ['18-24', '25-34', '35-44'], 
            n_users,
            p=[0.4, 0.4, 0.2]  # 偏向年轻
        )
        
        young_spending = np.random.choice(
            ['低', '中低', '中', '中高', '高'],
            n_users,
            p=[0.1, 0.2, 0.3, 0.3, 0.1]  # 偏向中高消费
        )
        
        # 群体2: 中年用户群
        middle_age_ranges = np.random.choice(
            ['25-34', '35-44', '45-54'], 
            n_users,
            p=[0.2, 0.4, 0.4]  # 偏向中年
        )
        
        middle_spending = np.random.choice(
            ['低', '中低', '中', '中高', '高'],
            n_users,
            p=[0.2, 0.3, 0.3, 0.1, 0.1]  # 偏向中低消费
        )
        
        # 共同的城市分布(但比例略有不同)
        cities1 = np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆'],
            n_users,
            p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]
        )
        
        cities2 = np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆'],
            n_users,
            p=[0.12, 0.12, 0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]
        )
        
        # 创建数据框
        cls.group1_data = pl.LazyFrame({
            'customer_id': range(1, n_users + 1),
            'age_range': young_age_ranges,
            'city': cities1,
            'category': np.random.choice(['服装', '数码', '美妆', '食品', '家居'], n_users),
            'spending_level': young_spending,
            'member_level': np.random.choice(['普通', '银卡', '金卡', '钻石'], n_users)
        })
        
        cls.group2_data = pl.LazyFrame({
            'customer_id': range(n_users + 1, 2 * n_users + 1),
            'age_range': middle_age_ranges,
            'city': cities2,
            'category': np.random.choice(['服装', '数码', '美妆', '食品', '家居'], n_users),
            'spending_level': middle_spending,
            'member_level': np.random.choice(['普通', '银卡', '金卡', '钻石'], n_users)
        })
        
        # 2. 创建特殊测试数据
        # 完全相同的群体
        cls.identical_group = cls.group1_data
        
        # 完全不同的群体(极端值)
        cls.extreme_group = pl.LazyFrame({
            'customer_id': range(1, 101),
            'age_range': ['55+'] * 100,
            'city': ['其他'] * 100,
            'category': ['其他'] * 100,
            'spending_level': ['特低'] * 100,
            'member_level': ['体验卡'] * 100
        })
        
        # 包含新类别的群体
        cls.new_categories_group = pl.LazyFrame({
            'customer_id': range(1, 101),
            'age_range': ['65+'] * 50 + ['18-24'] * 50,
            'city': ['新城市'] * 50 + ['北京'] * 50,
            'category': ['新品类'] * 50 + ['服装'] * 50,
            'spending_level': ['特高'] * 50 + ['高'] * 50,
            'member_level': ['黑金卡'] * 50 + ['钻石'] * 50
        })

    def test_basic_similarity(self):
        """测试基本相似度计算"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(self.group1_data, self.group2_data)
        
        # 验证结果
        self.assertGreater(result['similarity_score'], 0.5)  # 应该有一定相似度
        self.assertLess(result['similarity_score'], 0.9)     # 但不应该太相似
        
        # 验证详情
        self.assertEqual(len(result['details']['feature_details']), len(self.feature_cols))
        for detail in result['details']['feature_details']:
            self.assertLessEqual(len(detail['bins']), 10)  # 分箱数不超过设定值
            
    def test_identical_groups(self):
        """测试完全相同的群体"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(self.group1_data, self.identical_group)
        self.assertGreaterEqual(result['similarity_score'], 0.95)
        
    def test_different_groups(self):
        """测试完全不同的群体"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(self.group1_data, self.extreme_group)
        self.assertLess(result['similarity_score'], 0.3)
        
    def test_new_categories(self):
        """测试处理新类别"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=5  # 使用较小的分箱数测试类别合并
        )
        result = stage.forward(self.group1_data, self.new_categories_group)
        
        # 验证新类别被正确处理
        for detail in result['details']['feature_details']:
            if detail['feature'] in ['city', 'category', 'member_level']:
                self.assertIn('其他', detail['bins'])
                
    def test_different_bin_methods(self):
        """测试不同的分箱方法"""
        methods = ['equal_width', 'equal_freq', 'kmeans']
        results = {}
        
        for method in methods:
            stage = BinnedKLSimilarityStage(
                feature_cols=self.feature_cols,
                bin_method=method,
                n_bins=10
            )
            result = stage.forward(self.group1_data, self.group2_data)
            results[method] = result['similarity_score']
            
        # 验证不同方法产生的结果有所不同但都在合理范围内
        for method, score in results.items():
            self.assertGreater(score, 0.3)
            self.assertLess(score, 0.9)
            
    def test_outlier_handling(self):
        """测试异常值处理"""
        np.random.seed(42)
        n_users = 10000
        
        # 生成正常分布的数值特征
        order_amount = np.random.normal(100, 30, n_users)  # 均值100，标准差30
        order_count = np.random.normal(10, 3, n_users)     # 均值10，标准差3
        avg_price = np.random.normal(50, 15, n_users)      # 均值50，标准差15
        
        # 创建基准数据集
        base_data = pl.LazyFrame({
            'customer_id': range(1, n_users + 1),
            'order_amount': order_amount,
            'order_count': order_count,
            'avg_price': avg_price
        })
        
        # 创建带异常值的数据集
        # 为10%的数据添加极端值
        outlier_amount = np.where(
            np.random.random(n_users) < 0.1,
            np.random.normal(1000, 300, n_users),  # 异常值：均值1000
            np.random.normal(100, 30, n_users)     # 正常值：均值100
        )
        
        outlier_count = np.where(
            np.random.random(n_users) < 0.1,
            np.random.normal(100, 30, n_users),    # 异常值：均值100
            np.random.normal(10, 3, n_users)       # 正常值：均值10
        )
        
        outlier_price = np.where(
            np.random.random(n_users) < 0.1,
            np.random.normal(500, 150, n_users),   # 异常值：均值500
            np.random.normal(50, 15, n_users)      # 正常值：均值50
        )
        
        outlier_data = pl.LazyFrame({
            'customer_id': range(1, n_users + 1),
            'order_amount': outlier_amount,
            'order_count': outlier_count,
            'avg_price': outlier_price
        })
        
        # 对比有无异常值处理的结果
        stage_with_handling = BinnedKLSimilarityStage(
            feature_cols=['order_amount', 'order_count', 'avg_price'],
            handle_outliers=True,
            n_bins=10
        )
        stage_without_handling = BinnedKLSimilarityStage(
            feature_cols=['order_amount', 'order_count', 'avg_price'],
            handle_outliers=False,
            n_bins=10
        )
        
        result_with = stage_with_handling.forward(base_data, outlier_data)
        result_without = stage_without_handling.forward(base_data, outlier_data)
        
        # 打印详细信息以便调试
        print("\n异常值处理测试结果:")
        print(f"有异常值处理的相似度: {result_with['similarity_score']}")
        print(f"无异常值处理的相似度: {result_without['similarity_score']}")
        
        # 有异常值处理的相似度应该更高
        self.assertGreater(result_with['similarity_score'], 
                          result_without['similarity_score'])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
