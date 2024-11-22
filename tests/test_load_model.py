from stages.model import LoadModel
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
import json

def test_load_models():
    # 准备测试数据
    X, y = load_iris(return_X_y=True)
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # 保存XGBoost模型
    xgb_model = xgb.XGBClassifier(
        n_estimators=3,
        objective='multi:softmax',
        num_class=3
    )
    xgb_model.fit(X, y)
    
    xgb_path = 'test_xgb_model.json'
    # 保存模型和配置
    xgb_model.save_model(xgb_path)
    
    # 添加特征名到配置
    config = {
        'feature_names': feature_names,
        'learner': {
            'objective': 'multi:softmax'
        }
    }
    with open(xgb_path, 'r') as f:
        model_config = json.load(f)
    model_config.update(config)
    with open(xgb_path, 'w') as f:
        json.dump(model_config, f)
    
    # 保存决策树模型
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X, y)
    dt_path = 'test_dt_model.pkl'
    with open(dt_path, 'wb') as f:
        pickle.dump(dt_model, f)
    
    try:
        # 测试加载XGBoost模型
        xgb_loader = LoadModel(xgb_path, 'XGB')
        xgb_result = xgb_loader.forward()
        print("XGBoost模型加载成功")
        print(f"模型类型: {xgb_result['type']}")
        print(f"特征名: {xgb_result['feature_names']}")
        
        # 测试加载决策树模型
        dt_loader = LoadModel(dt_path, 'DT')
        dt_result = dt_loader.forward()
        print("\n决策树模型加载成功")
        print(f"模型类型: {dt_result['type']}")
        print(f"特征名: {dt_result['feature_names']}")
        
    finally:
        # 清理测试文件
        if os.path.exists(xgb_path):
            os.remove(xgb_path)
        if os.path.exists(dt_path):
            os.remove(dt_path)

if __name__ == "__main__":
    test_load_models() 