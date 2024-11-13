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
            p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]  # ���线和新一线城市分布
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
    #         'city': ['北京'] * 6 + ['���海'] * 4,  # 相似城市分布
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
        """复用TestCustomerSimilarityStage的测试数据准备逻辑"""
        TestCustomerSimilarityStage.setUpClass()
        cls.feature_cols = TestCustomerSimilarityStage.feature_cols
        cls.group1_data = TestCustomerSimilarityStage.group1_data
        cls.group2_data = TestCustomerSimilarityStage.group2_data
        cls.empty_data = TestCustomerSimilarityStage.empty_data
        cls.single_record_data = TestCustomerSimilarityStage.single_record_data
        cls.data_with_nulls = TestCustomerSimilarityStage.data_with_nulls
        cls.extreme_data = TestCustomerSimilarityStage.extreme_data

    def test_different_binning_methods(self):
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
            
        # 验证所有方法都产生了有效的相似度分数
        for method, score in results.items():
            self.assertGreaterEqual(score, 0.0, f"{method}方法产生的相似度小于0")
            self.assertLessEqual(score, 1.0, f"{method}方法产生的相似度大于1")
            
        # 验证不同方法产生的结果有所不同
        self.assertNotEqual(results['equal_width'], results['equal_freq'])
        self.assertNotEqual(results['equal_width'], results['kmeans'])
        self.assertNotEqual(results['equal_freq'], results['kmeans'])

    def test_bin_numbers(self):
        """测试不同的分箱数量"""
        bin_numbers = [5, 10, 20]
        results = {}
        
        for n_bins in bin_numbers:
            stage = BinnedKLSimilarityStage(
                feature_cols=self.feature_cols,
                bin_method='equal_freq',
                n_bins=n_bins
            )
            result = stage.forward(self.group1_data, self.group2_data)
            results[n_bins] = result['similarity_score']
            
        # 验证分箱数量的影响
        for n_bins, score in results.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
        # 验证分箱数量变化确实影响了结果
        self.assertNotEqual(results[5], results[20])

    def test_identical_groups(self):
        """测试完全相同的群体"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(self.group1_data, self.group1_data)
        
        # 相同群体的相似度应该非常接近1
        self.assertGreaterEqual(result['similarity_score'], 0.95)

    def test_partially_similar_groups(self):
        """测试部分相似的群体"""
        # 创建两个部分相似的群体
        base_data = {
            'customer_id': range(1, 101),
            'value1': np.concatenate([
                np.random.normal(0, 1, 50),  # 前50个样本分布相同
                np.random.normal(0, 1, 50)   # 后50个样本分布相同
            ]),
            'value2': np.concatenate([
                np.random.normal(0, 1, 50),  # 前50个样本分布相同
                np.random.normal(3, 1, 50)   # 后50个样本分布不同
            ])
        }
        
        group1 = pl.LazyFrame(base_data)
        
        group2_data = {
            'customer_id': range(101, 201),
            'value1': np.concatenate([
                np.random.normal(0, 1, 50),  # 前50个样本分布相同
                np.random.normal(0, 1, 50)   # 后50个样本分布相同
            ]),
            'value2': np.concatenate([
                np.random.normal(0, 1, 50),  # 前50个样本分布相同
                np.random.normal(5, 1, 50)   # 后50个样本分布更不同
            ])
        }
        
        group2 = pl.LazyFrame(group2_data)
        
        stage = BinnedKLSimilarityStage(
            feature_cols=['value1', 'value2'],
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(group1, group2)
        
        # 部分相似群体的相似度应该在中等水平
        self.assertGreater(result['similarity_score'], 0.4)
        self.assertLess(result['similarity_score'], 0.8)
        
        # 验证特征级别的相似度
        feature_scores = {
            detail['feature']: detail['kl_divergence'] 
            for detail in result['details']['feature_details']
        }
        # value1应该有更高的相似度（更低的KL散度）
        self.assertLess(feature_scores['value1'], feature_scores['value2'])

    def test_subset_superset_groups(self):
        """测试子集和全集关系的群体"""
        # 创建一个基础群体
        np.random.seed(42)
        base_distribution = np.concatenate([
            np.random.normal(0, 1, 700),    # 主要分布
            np.random.normal(3, 0.5, 200),  # 次要分布
            np.random.normal(6, 0.5, 100)   # 小众分布
        ])
        
        # 创建全集
        superset = pl.LazyFrame({
            'customer_id': range(1, len(base_distribution) + 1),
            'value': base_distribution
        })
        
        # 创建子集（只包含主要和次要分布）
        subset = pl.LazyFrame({
            'customer_id': range(1, 801),
            'value': base_distribution[:800]
        })
        
        stage = BinnedKLSimilarityStage(
            feature_cols=['value'],
            bin_method='equal_freq',
            n_bins=10
        )
        
        # 测试子集到全集的相似度
        result = stage.forward(subset, superset)
        subset_to_superset_similarity = result['similarity_score']
        
        # 测试全集到子集的相似度
        result = stage.forward(superset, subset)
        superset_to_subset_similarity = result['similarity_score']
        
        # 验证相似度在合理范围内
        self.assertGreater(subset_to_superset_similarity, 0.7)
        self.assertGreater(superset_to_subset_similarity, 0.7)
        
        # 验证对称性（两个方向的相似度应该接近）
        self.assertAlmostEqual(
            subset_to_superset_similarity,
            superset_to_subset_similarity,
            places=2
        )


    def test_completely_different_groups(self):
        """测试完全不同的群体"""
        # 创建两个完全不同的数值特征群体
        group1 = pl.LazyFrame({
            'customer_id': range(1, 101),
            'value1': np.random.normal(0, 1, 100),  # 均值为0的正态分布
            'value2': np.random.normal(0, 1, 100)
        })
        
        group2 = pl.LazyFrame({
            'customer_id': range(101, 201),
            'value1': np.random.normal(5, 1, 100),  # 均值为5的正态分布
            'value2': np.random.normal(5, 1, 100)
        })
        
        stage = BinnedKLSimilarityStage(
            feature_cols=['value1', 'value2'],
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(group1, group2)
        
        # 完全不同群体的相似度应该很低
        self.assertLessEqual(result['similarity_score'], 0.3)

    def test_overlapping_groups(self):
        """测试有重叠部分的群体"""
        np.random.seed(42)
        
        # 创建三个不同的分布
        dist1 = np.random.normal(0, 1, 300)    # 分布1
        dist2 = np.random.normal(3, 1, 300)    # 分布2
        dist3 = np.random.normal(6, 1, 300)    # 分布3
        
        # 创建两个重叠的群体
        group1_data = {
            'customer_id': range(1, 601),
            'value': np.concatenate([dist1, dist2])  # 包含分布1和2
        }
        
        group2_data = {
            'customer_id': range(601, 1201),
            'value': np.concatenate([dist2, dist3])  # 包含分布2和3
        }
        
        group1 = pl.LazyFrame(group1_data)
        group2 = pl.LazyFrame(group2_data)
        
        stage = BinnedKLSimilarityStage(
            feature_cols=['value'],
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(group1, group2)
        
        # 验证重叠群体的相似度
        similarity = result['similarity_score']
        
        # 由于有1/2的重叠，相似度应该在中等水平
        self.assertGreater(similarity, 0.3)
        self.assertLess(similarity, 0.7)
        
        # 测试分布细节
        feature_detail = result['details']['feature_details'][0]
        self.assertEqual(len(feature_detail['bins']), stage.n_bins + 1)
        self.assertEqual(len(feature_detail['dist1']), stage.n_bins)
        self.assertEqual(len(feature_detail['dist2']), stage.n_bins)

    def test_smooth_factor(self):
        """测试平滑因子的影响"""
        # 使用不同的平滑因子
        smooth_factors = [1e-10, 1e-5, 1e-3]
        results = {}
        
        for factor in smooth_factors:
            stage = BinnedKLSimilarityStage(
                feature_cols=self.feature_cols,
                smooth_factor=factor
            )
            result = stage.forward(self.group1_data, self.group2_data)
            results[factor] = result['similarity_score']
            
        # 验证平滑因子的影响
        for factor, score in results.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
        # 验证不同平滑因子产生的结果有细微差异
        self.assertNotEqual(results[1e-10], results[1e-3])

    def test_feature_details(self):
        """测试特征详情输出"""
        stage = BinnedKLSimilarityStage(
            feature_cols=self.feature_cols,
            bin_method='equal_freq',
            n_bins=10
        )
        result = stage.forward(self.group1_data, self.group2_data)
        
        # 验证结果包含所有必要的详细信息
        self.assertIn('feature_details', result['details'])
        
        for feature_detail in result['details']['feature_details']:
            self.assertIn('feature', feature_detail)
            self.assertIn('kl_divergence', feature_detail)
            self.assertIn('bins', feature_detail)
            self.assertIn('dist1', feature_detail)
            self.assertIn('dist2', feature_detail)
            
            # 验证分布的有效性
            self.assertEqual(len(feature_detail['dist1']), stage.n_bins)
            self.assertEqual(len(feature_detail['dist2']), stage.n_bins)
            self.assertAlmostEqual(sum(feature_detail['dist1']), 1.0, places=5)
            self.assertAlmostEqual(sum(feature_detail['dist2']), 1.0, places=5)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
