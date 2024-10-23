import unittest
import numpy as np
import polars as pl
from stages.xgb import XGBImportance

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
        })

        self.model = XGBImportance(label_col=self.label_col, train_cols=self.train_cols,
                                   importance_type="weight", topK=2)

    def test_forward(self):
        transformed_data, out_features = self.model.forward(self.data)

        # 验证输出特征
        self.assertEqual(len(out_features), 2)  # 应该返回两个特征
        self.assertIn("feature1", out_features)
        self.assertIn("feature2", out_features)

        # 验证转换后的数据
        self.assertIsInstance(transformed_data, pl.LazyFrame)
        self.assertEqual(transformed_data.columns, out_features)

    def test_forward_with_model_xgb_f_importance(self):
        model_xgb_f_importance = ["feature1", "feature2"]
        transformed_data, out_features = self.model.forward(self.data, model_xgb_f_importance)

        # 验证输出特征
        self.assertEqual(out_features, model_xgb_f_importance)

        # 验证转换后的数据
        self.assertIsInstance(transformed_data, pl.LazyFrame)
        self.assertEqual(transformed_data.columns, out_features)

if __name__ == '__main__':
    unittest.main()
