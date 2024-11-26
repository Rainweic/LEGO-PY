import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from stages.model import ConvertXGBToSQL
import json

if __name__ == "__main__":
    # 加载数据
    X, y = load_iris(return_X_y=True)
    y = np.where(y > 0, 0, 1)
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df = pl.DataFrame(X, schema=feature_names)
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

    # 训练二分类模型
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        # 'base_score': 0.5,  # 显式设置base_score
        'learning_rate': 0.3  # 显式设置学习率
    }
    binary_model = xgb.train(params, dtrain, num_boost_round=1)
    
    # 打印模型信息
    print("\n=== 模型配置 ===")
    config = json.loads(binary_model.save_config())
    base_score = float(config['learner']['learner_model_param']['base_score'])
    learning_rate = float(config['learner']['gradient_booster']['tree_train_param']['learning_rate'])
    print(f"Base score: {base_score}")
    print(f"Learning rate: {learning_rate}")
    
    # 获取树结构
    tree_str = binary_model.get_dump()[0]
    print("\nTree structure:")
    print(tree_str)
    tree = json.loads(binary_model.get_dump(dump_format='json')[0])
    
    # 打印叶子节点值
    print("\nLeaf values:")
    if 'children' in tree:
        print(f"Left leaf: {tree['children'][0]['leaf']}")
        print(f"Right leaf: {tree['children'][1]['leaf']}")
    
    # 转换二分类模型
    binary_data = {
        'model': binary_model,
        'cols': feature_names
    }
    
    converter = ConvertXGBToSQL()
    print("\n=== 二分类模型 SQL ===")
    binary_sql = converter.forward(binary_data)
    print(binary_sql['model'])

    # 比较预测结果
    y_score = binary_model.predict(dtrain)
    y_margin = binary_model.predict(dtrain, output_margin=True)  # 获取原始margin值
    y_score_sql = df.sql(binary_sql['model']).select('probability').to_series()

    print("\n=== 预测结果比较 ===")
    print("原始预测结果:", y_score[:5])
    print("原始margin值:", y_margin[:5])
    print("SQL预测结果:", y_score_sql.head())
    
    # 计算中间值进行比较
    print("\n=== 中间计算值 ===")
    # 获取树的预测值
    tree_value = np.where(X[:, 2] < 3.0, tree['children'][0]['leaf'], tree['children'][1]['leaf'])
    # 应用学习率
    tree_value = tree_value * learning_rate
    # 计算最终的margin
    margin = base_score + tree_value
    # 计算概率
    prob = 1 / (1 + np.exp(-margin))
    print("手动计算:")
    print("- 树的原始值:", tree_value[:5])
    print("- Margin值:", margin[:5])
    print("- 最终概率:", prob[:5])
    
    # 打印差异
    print("\n=== 差异分析 ===")
    print("XGB vs SQL差异:", np.abs(y_score - y_score_sql.to_numpy())[:5])
    print("XGB margin vs 手动margin差异:", np.abs(y_margin - margin)[:5])
    print("XGB prob vs 手动prob差异:", np.abs(y_score - prob)[:5])
    
    # 尝试不同的计算方式
    print("\n=== 其他计算方式 ===")
    prob1 = 1 / (1 + np.exp(-(tree_value)))  # 不使用base_score
    prob2 = 1 / (1 + np.exp(-margin * learning_rate))  # 对整个margin应用学习率
    print("方式1 (无base_score):", prob1[:5])
    print("方式2 (margin*lr):", prob2[:5])
    
    np.testing.assert_array_almost_equal(
        y_score, 
        y_score_sql.to_numpy(),
        decimal=5,
        err_msg='二分类模型预测结果不一致'
    )