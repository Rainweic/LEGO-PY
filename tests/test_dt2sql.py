from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from stages.model import ConvertDTToSQL

if __name__ == "__main__":
    # 加载鸢尾花数据集
    X, y = load_iris(return_X_y=True)
    
    # 获取正确的特征名和类别名
    iris = load_iris()
    feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    class_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']
    
    # 训练决策树
    m = DecisionTreeClassifier(max_depth=2)
    m.fit(X, y)

    # 转换为SQL
    model_data = {
        'model': m,
        'feature_names': feature_names,  # 使用正确的特征名
        'class_names': class_names,      # 使用正确的类别名
        'table_prefix': 'iris'           # 可选
    }

    converter = ConvertDTToSQL()
    sql = converter.forward(model_data)
    print("生成的SQL语句：")
    print(sql['model'])  # 因为forward返回的是字典
