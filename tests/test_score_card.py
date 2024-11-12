import unittest
import polars as pl
import numpy as np
from stages.score_card import ScoreCard
import os
import shutil


class TestScoreCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的准备工作"""
        # 创建测试数据
        np.random.seed(42)
        n_samples = 1000
        
        # 生成特征
        age = np.random.normal(35, 10, n_samples)
        income = np.random.normal(50000, 20000, n_samples)
        education = np.random.choice([1, 2, 3, 4], n_samples)
        
        # 生成目标变量（使其与特征相关）
        prob = 1 / (1 + np.exp(-(0.03 * age + 0.00001 * income + 0.5 * education)))
        target = np.random.binomial(1, prob)
        
        # 转换为WOE值（这里简单模拟WOE转换后的值）
        age_woe = (age - np.mean(age)) / np.std(age)
        income_woe = (income - np.mean(income)) / np.std(income)
        education_woe = (education - np.mean(education)) / np.std(education)
        
        # 创建训练集
        cls.train_data = pl.DataFrame({
            'age_woe': age_woe[:800],
            'income_woe': income_woe[:800],
            'education_woe': education_woe[:800],
            'target': target[:800]
        }).lazy()
        
        # 创建测试集
        cls.test_data = pl.DataFrame({
            'age_woe': age_woe[800:],
            'income_woe': income_woe[800:],
            'education_woe': education_woe[800:],
            'target': target[800:]
        }).lazy()
        
        # 创建缓存目录
        os.makedirs('cache/test_job/models', exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """测试结束后的清理工作"""
        # 删除测试过程中创建的文件
        if os.path.exists('cache/test_job'):
            shutil.rmtree('cache/test_job')
    
    def setUp(self):
        """每个测试用例开始前的准备"""
        self.scorecard = ScoreCard(
            features=['age_woe', 'income_woe', 'education_woe'],
            label_col='target',
            train_params={
                'C': 0.1,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            }
        )
        self.scorecard.job_id = 'test_job'
    
    def test_forward(self):
        """测试模型训练和预测"""
        # 训练模型
        result = self.scorecard.forward(self.train_data, self.test_data)

    
    
    

if __name__ == '__main__':
    unittest.main()
