import unittest
import polars as pl
import numpy as np
from stages.score_card import ScoreCard
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class TestScoreCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试开始前的准备工作"""
        try:
            # 尝试从UCI下载German Credit数据
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            column_names = [
                'status', 'duration', 'credit_history', 'purpose', 'amount',
                'savings', 'employment', 'installment_rate', 'personal_status_sex',
                'other_debtors', 'residence_since', 'property', 'age',
                'other_installment_plans', 'housing', 'existing_credits',
                'job', 'num_dependents', 'own_telephone', 'foreign_worker',
                'class'
            ]
            data = pd.read_csv(url, sep=' ', names=column_names)
            
            # 将目标变量转换为0/1 (1=bad, 2=good -> 0=good, 1=bad)
            y = (data['class'] == 1).astype(int)
            X = data.drop('class', axis=1)
            
            # 数值化处理
            X = pd.get_dummies(X, drop_first=True)
            
            # 数值特征标准化（模拟WOE转换）
            numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y
            )
            
            # 转换为LazyFrame
            cls.train_data = pl.DataFrame({
                **{col: X_train[col].values for col in X_train.columns},
                'target': y_train
            }).lazy()
            
            cls.test_data = pl.DataFrame({
                **{col: X_test[col].values for col in X_test.columns},
                'target': y_test
            }).lazy()
            
            # 记录特征列表
            cls.feature_cols = list(X_train.columns)
            
            print(f"数据集加载完成:")
            print(f"特征数量: {len(cls.feature_cols)}")
            print(f"训练集样本数: {len(y_train)}")
            print(f"测试集样本数: {len(y_test)}")
            print(f"正样本比例: {y.mean():.2%}")
            
        except Exception as e:
            print(f"加载German Credit数据集失败: {str(e)}")
            # 使用备用的模拟数据
            cls._create_mock_data()
    
    @classmethod
    def _create_mock_data(cls):
        """创建模拟数据（当无法加载真实数据集时使用）"""
        print("使用模拟数据集")
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # 生成特征
        X = np.random.randn(n_samples, n_features)
        
        # 生成目标变量
        w = np.random.randn(n_features)
        z = X @ w
        prob = 1 / (1 + np.exp(-z))
        y = np.random.binomial(1, prob)
        
        # 特征名称
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        cls.feature_cols = feature_cols
        
        # 分割训练集和测试集
        train_size = int(0.8 * n_samples)
        
        # 创建训练集
        cls.train_data = pl.DataFrame({
            **{col: X[:train_size, i] for i, col in enumerate(feature_cols)},
            'target': y[:train_size]
        }).lazy()
        
        # 创建测试集
        cls.test_data = pl.DataFrame({
            **{col: X[train_size:, i] for i, col in enumerate(feature_cols)},
            'target': y[train_size:]
        }).lazy()
    
    def setUp(self):
        """每个测试用例开始前的准备"""
        self.scorecard = ScoreCard(
            features=self.feature_cols,
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
        
        # 检查模型参数
        self.assertIn('weight', result)
        self.assertIn('bias', result)
        self.assertIn('features', result)
        self.assertIn('score_params', result)
        
        # 检查特征权重
        self.assertEqual(len(result['weight'][0]), len(self.feature_cols))
        
        # 检查评估指标
        self.assertIn('train', self.scorecard.metrics)
        self.assertIn('eval', self.scorecard.metrics)
        
        # 打印模型性能
        print("\n模型评估结果:")
        print("训练集:")
        for metric, value in self.scorecard.metrics['train'].items():
            print(f"  {metric}: {value:.4f}")
        print("测试集:")
        for metric, value in self.scorecard.metrics['eval'].items():
            print(f"  {metric}: {value:.4f}")
    
    def test_predict(self):
        """测试预测功能"""
        # 先训练模型
        model_info = self.scorecard.forward(self.train_data, self.test_data)
        
        # 使用模型进行预测
        predictions = ScoreCard.predict(model_info, self.test_data)
        result_df = predictions.collect()
        
        # 检查预测结果
        self.assertIn('y_score', result_df.columns)
        self.assertIn('ScoreCard_Score', result_df.columns)
        
        # 检查预测值范围
        y_scores = result_df['y_score'].to_numpy()
        self.assertTrue(np.all((y_scores >= 0) & (y_scores <= 1)))
        
        # 打印预测结果统计
        scores = result_df['ScoreCard_Score'].to_numpy()
        print("\n预测分数统计:")
        print(f"最小分数: {np.min(scores)}")
        print(f"最大分数: {np.max(scores)}")
        print(f"平均分数: {np.mean(scores):.2f}")
        print(f"分数标准差: {np.std(scores):.2f}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
