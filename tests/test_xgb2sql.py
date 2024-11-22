import xgboost as xgb
from sklearn.datasets import load_iris
from stages.model import ConvertXGBToSQL

if __name__ == "__main__":
    # 加载数据
    X, y = load_iris(return_X_y=True)
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # 训练二分类模型
    binary_y = (y == 0).astype(int)  # 将问题转化为二分类
    binary_model = xgb.XGBClassifier(
        n_estimators=3,
        max_depth=3,
        objective='binary:logistic'
    )
    binary_model.fit(X, binary_y)
    
    # 转换二分类模型
    binary_data = {
        'model': binary_model,
        'feature_names': feature_names,
        'table_prefix': 'iris'
    }
    
    converter = ConvertXGBToSQL()
    print("=== 二分类模型 SQL ===")
    binary_sql = converter.forward(binary_data)
    print(binary_sql['model'])
    print("\n")
    
    # 训练多分类模型
    multi_model = xgb.XGBClassifier(
        n_estimators=3,
        max_depth=3,
        objective='multi:softmax',
        num_class=3
    )
    multi_model.fit(X, y)
    
    # 转换多分类模型
    multi_data = {
        'model': multi_model,
        'feature_names': feature_names,
        'table_prefix': 'iris'
    }
    
    print("=== 多分类模型 SQL ===")
    multi_sql = converter.forward(multi_data)
    print(multi_sql['model'])
