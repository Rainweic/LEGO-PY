import unittest
import numpy as np
import polars as pl
from stages.xgb import XGB
import xgboost as xgb

class TestXGB(unittest.TestCase):

    def setUp(self):
        # 创建测试数据
        self.label_col = "label"
        self.train_cols = ["feature1", "feature2"]
        
        np.random.seed(0)  # 设置随机种子以便复现
        num_samples = 1000  # 增加样本数量以获得更稳定的结果
        
        self.train_data = pl.DataFrame({
            "feature1": np.random.rand(num_samples) * 100,
            "feature2": np.random.rand(num_samples) * 100,
            "label": np.random.randint(0, 2, size=num_samples)
        })

        self.eval_data = pl.DataFrame({
            "feature1": np.random.rand(num_samples // 2) * 100,
            "feature2": np.random.rand(num_samples // 2) * 100,
            "label": np.random.randint(0, 2, size=num_samples // 2)
        })

        self.model = XGB(label_col=self.label_col, train_cols=self.train_cols, num_round=10)

    def test_train(self):
        trained_model = self.model.train(self.train_data)
        
        # 验证返回的是否为 XGBoost 模型
        self.assertIsInstance(trained_model['xgb'], xgb.Booster)

    def test_train_with_eval(self):
        trained_model = self.model.train(self.train_data, self.eval_data)
        
        # 验证返回的是否为 XGBoost 模型
        self.assertIsInstance(trained_model['xgb'], xgb.Booster)

    def test_forward(self):
        result = self.model.forward(self.train_data, self.eval_data)
        
        # 验证返回的是否为 XGBoost 模型
        self.assertIsInstance(result['xgb'], xgb.Booster)

    def test_train_cols_auto_detection(self):
        model = XGB(label_col=self.label_col)  # 不指定 train_cols
        trained_model = model.train(self.train_data)
        
        # 验证是否正确检测到训练列
        self.assertEqual(set(model.train_cols), set(self.train_cols))

    def test_label_col_exclusion(self):
        train_data_with_label = self.train_data.clone()
        train_data_with_label = train_data_with_label.with_columns(pl.col("label").alias("label_copy"))
        
        model = XGB(label_col="label_copy", train_cols=["feature1", "feature2", "label_copy"])
        model.train(train_data_with_label)
        
        # 验证 label 列是否被正确排除
        self.assertNotIn("label_copy", model.train_cols)

    def test_custom_params(self):
        custom_params = {
            'max_depth': 3,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        model = XGB(label_col=self.label_col, train_cols=self.train_cols, train_params=custom_params)
        trained_model = model.train(self.train_data)
        
        # 验证自定义参数是否被正确使用
        self.assertEqual(model.train_params, custom_params)
    
    def test_save_load_model(self):
        model_dict = self.model.forward(self.train_data, self.eval_data)
        import pickle
        model_pk = pickle.dumps(model_dict)
        model_dict = pickle.loads(model_pk)
        out = self.model.predict(model_dict, self.eval_data)
        print("============================")
        print("预测结果", out)

if __name__ == '__main__':
    unittest.main()