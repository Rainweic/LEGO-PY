import unittest
import numpy as np
import polars as pl
from stages.importance import XGBImportance

class TestXGBImportance(unittest.TestCase):

    def setUp(self):
        # 创建测试数据
        self.label_col = "label"
        self.train_cols = ["feature1", "feature2", "label"]
        
        np.random.seed(0)  # 设置随机种子以便复现
        num_samples = 100  # 增加样本数量
        
        self.data = pl.DataFrame({
            "feature1": np.random.rand(num_samples) * 100,  # 随机生成特征1
            "feature2": np.random.rand(num_samples) * 100,  # 随机生成特征2
            "label": np.random.randint(0, 2, size=num_samples)  # 随机生成二分类标签
        }).lazy()

        # 创建转换数据
        self.transform_data = pl.DataFrame({
            "feature1": np.random.rand(num_samples) * 10000000,
            "feature2": np.random.rand(num_samples) * 10000000
        }).lazy()

        self.model = XGBImportance(label_col=self.label_col, train_cols=self.train_cols,
                                   importance_type="weight")

    def test_forward(self):
        out_features, transformed_data = self.model.forward(self.data)

        # 验证输出特征
        self.assertEqual(len(out_features), 2)  # 应该返回两个特征
        self.assertIn("feature1", out_features)
        self.assertIn("feature2", out_features)

        # 验证转换后的数据
        self.assertIsNone(transformed_data)

    def test_forward_with_transformed_data(self):
        out_features, transformed_data = self.model.forward(self.data, self.transform_data)

        # 验证输出特征
        self.assertEqual(len(out_features), 2)  # 应该返回两个特征
        self.assertIn("feature1", out_features)
        self.assertIn("feature2", out_features)

        # 验证转换后的数据
        self.assertIsNotNone(transformed_data)
        self.assertEqual(transformed_data.columns, out_features)

if __name__ == '__main__':
    unittest.main()