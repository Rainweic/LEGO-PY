import os
import uuid
from dags.stage import CustomStage
from sklearn.tree import _tree
import numpy as np
import json
import pickle
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class LoadModel(CustomStage):
    """加载模型"""

    def __init__(self, file_path, model_type):
        """
        Args:
            file_path: 模型文件路径
            model_type: 模型类型，支持 'XGB', 'DT', 'RF'
        """
        super().__init__(n_outputs=1)
        self.file_path = file_path
        self.model_type = model_type.upper()

    def _load_xgboost(self):
        """加载XGBoost模型"""
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.json':
            # 加载JSON格式
            with open(self.file_path, 'r') as f:
                config = json.load(f)
                
            # 根据配置确定是分类还是回归
            objective = config.get('learner', {}).get('objective', '')
            if 'binary:' in objective or 'multi:' in objective:
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            
            model.load_model(self.file_path)
            
            # 提取特征名
            feature_names = config.get('feature_names', None)
            
        elif file_ext == '.bin':
            # 尝试加载为分类器
            try:
                model = xgb.XGBClassifier()
                model.load_model(self.file_path)
            except Exception:
                # 如果失败，尝试加载为回归器
                model = xgb.XGBRegressor()
                model.load_model(self.file_path)
            
            # 尝试从模型中获取特征名
            feature_names = getattr(model, 'feature_names_in_', None)
            
        else:
            raise ValueError(f"不支持的XGBoost模型文件格式: {file_ext}")
            
        return model, feature_names

    def _load_sklearn(self):
        """加载sklearn模型（决策树或随机森林）"""
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.pkl':
            # 加载pickle格式
            with open(self.file_path, 'rb') as f:
                model = pickle.load(f)
                
            # 验证模型类型
            valid_types = (
                DecisionTreeClassifier, DecisionTreeRegressor,
                RandomForestClassifier, RandomForestRegressor
            )
            
            if not isinstance(model, valid_types):
                raise ValueError(f"不支持的sklearn模型类型: {type(model)}")
                
            # 尝试从模型中获取特征名
            feature_names = getattr(model, 'feature_names_in_', None)
            
        else:
            raise ValueError(f"不支持的sklearn模型文件格式: {file_ext}")
            
        return model, feature_names

    def forward(self):
        """读取模型文件，返回模型字典
        
        Returns:
            dict: {
                'model': 加载的模型对象,
                'type': 模型类型,
                'feature_names': 特征名列表（如果有）
            }
        
        Raises:
            ValueError: 当模型类型或文件格式不支持时
        """
        self.logger.info(f"开始加载{self.model_type}模型: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"模型文件不存在: {self.file_path}")
            
        try:
            if self.model_type == 'XGB':
                model, feature_names = self._load_xgboost()
            elif self.model_type in ['DT', 'RF']:
                model, feature_names = self._load_sklearn()
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
            result = {
                'model': model,
                'type': self.model_type,
                'feature_names': feature_names
            }
            
            self.logger.info(f"模型加载成功")
            if feature_names:
                self.logger.info(f"特征名: {feature_names}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise


class ExportModel(CustomStage):

    def __init__(self):
        super().__init__(n_outputs=0)

    def forward(self, model):
    
        model_type = model['type']
        model = model['model']

        # 保存到临时文件目录
        model_name = f"{model_type}_{uuid.uuid4()}.bin"
        model_dir = os.path.join('cache', self.job_id, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(model_dir, model_name)

        self.logger.info(f"模型将保存到: {model_name}")

        if model_type == 'XGB':
            model.save_model(model_name)


class ConvertDTToSQL(CustomStage):
    """将Sklearn训练的决策树转换成SQL语句"""

    def __init__(self):
        super().__init__(n_outputs=1)

    def _tree_to_sql(self, tree, feature_names, class_names=None, prefix=""):
        """递归将决策树转换为SQL CASE语句
        
        Args:
            tree: sklearn的决策树对象
            feature_names: 特征名列表
            class_names: 类别名列表（分类树用）
            prefix: SQL中的表前缀
        """
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth, sql_parts):
            indent = "  " * depth
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # 非叶子节点
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # 处理特征名
                if prefix:
                    name = f"{prefix}.{name}"
                
                sql_parts.append(f"{indent}CASE")
                sql_parts.append(f"{indent}WHEN {name} <= {threshold:.6f} THEN")
                
                # 递归处理左子树
                recurse(tree_.children_left[node], depth + 1, sql_parts)
                
                sql_parts.append(f"{indent}ELSE")
                
                # 递归处理右子树
                recurse(tree_.children_right[node], depth + 1, sql_parts)
                
                sql_parts.append(f"{indent}END")
                
            else:
                # 叶子节点
                if tree_.n_outputs == 1:
                    value = tree_.value[node][0]
                    if tree_.n_classes[0] == 1:
                        # 回归树
                        sql_parts.append(f"{indent}{value[0]:.6f}")
                    else:
                        # 分类树
                        if class_names is not None:
                            class_name = class_names[np.argmax(value)]
                        else:
                            class_name = np.argmax(value)
                        sql_parts.append(f"{indent}'{class_name}'")
                else:
                    # 多输出树
                    sql_parts.append(f"{indent}{tree_.value[node].tolist()}")
            
            return sql_parts

        sql_parts = []
        sql = "\n".join(recurse(0, 1, sql_parts))
        return sql

    def forward(self, model):
        """将决策树模型转换为SQL语句
        
        Args:
            model: 包含模型的字典，格式为 {'model': sklearn_model, 'feature_names': [...], 'class_names': [...]}
        """
        tree_model = model['model']
        feature_names = model.get('feature_names', None)
        class_names = model.get('class_names', None)
        prefix = model.get('table_prefix', '')
        
        # 如果没有提供特征名，使用默认的特征名 (feature_0, feature_1, ...)
        if feature_names is None:
            n_features = tree_model.n_features_in_
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # 转换为SQL
        sql = self._tree_to_sql(tree_model, feature_names, class_names, prefix)
        
        # 添加SELECT语句
        final_sql = f"SELECT\n{sql}\nAS prediction"
        
        self.logger.info("决策树已转换为SQL语句")
        self.logger.info(final_sql)
        return {"type": "SQL", "model": final_sql}
        

class ConvertXGBToSQL(CustomStage):
    """将XGBoost训练的模型转换为SQL语句"""

    def __init__(self, table_name: str = 'self', keep_cols: list = []):
        super().__init__(n_outputs=1)
        self.table_name = table_name
        self.keep_cols = keep_cols

    def _tree_to_sql(self, tree, feature_names, tree_index=0):
        """将单棵XGBoost树转换为SQL CASE语句"""
        def recurse(node, depth, sql_parts):
            indent = "  " * depth
            
            # 检查是否是叶子节点
            if 'leaf' in node:
                leaf_value = float(node['leaf'])
                sql_parts.append(f"{indent}{leaf_value:.6f}")
                return sql_parts
            
            # 非叶子节点
            split_feature = node['split']
            if isinstance(split_feature, int):
                feature_name = feature_names[split_feature]
            else:
                feature_name = split_feature
                
            threshold = float(node['split_condition'])
            
            sql_parts.append(f"{indent}CASE")
            # 参考开源库实现，处理missing值
            sql_parts.append(f"{indent}WHEN {feature_name} IS NULL THEN")
            if node.get('missing', 1) == 0:  # missing值走左子树
                if 'children' in node:
                    recurse(node['children'][0], depth + 1, sql_parts)
            else:  # missing值走右子树
                if 'children' in node:
                    recurse(node['children'][1], depth + 1, sql_parts)
            
            sql_parts.append(f"{indent}WHEN {feature_name} < {threshold:.6f} THEN")
            # 递归处理左子树
            if 'children' in node:
                recurse(node['children'][0], depth + 1, sql_parts)
                
            sql_parts.append(f"{indent}ELSE")
            # 递归处理右子树
            if 'children' in node:
                recurse(node['children'][1], depth + 1, sql_parts)
            
            sql_parts.append(f"{indent}END")
            return sql_parts
            
        sql_parts = []
        sql = "\n".join(recurse(tree, 1, sql_parts))
        return sql

    def forward(self, model):
        """将XGBoost模型转换为SQL语句"""
        xgb_model = model['model']
        feature_names = model.get('cols', [])

        self.logger.info(f"需要保留的列名: {self.keep_cols}")
        
        # 获取模型的JSON表示
        if isinstance(xgb_model, xgb.Booster):
            booster = xgb_model
            config = json.loads(booster.save_config())
            base_score = float(config['learner']['learner_model_param']['base_score'])
        else:
            booster = xgb_model.get_booster()
            base_score = getattr(xgb_model, 'base_score_', 0.5)
        
        # 计算base_score的对数几率
        self.logger.info(f"base_score: {base_score}")
        
        model_dump = booster.get_dump(dump_format='json')
        trees = [json.loads(tree_str) for tree_str in model_dump]
        
        # 转换每棵树
        tree_sqls = []
        for i, tree in enumerate(trees):
            tree_sql = self._tree_to_sql(tree, feature_names, i)
            tree_sqls.append(f"({tree_sql})")
        
        # 修改SQL计算逻辑：
        # 1. 计算base_score的对数几率
        # 2. 直接将树的结果相加（不需要学习率）
        # 3. 应用sigmoid函数得到最终概率
        final_sql = f"""
WITH tree_scores AS (
    SELECT 
        {", ".join(self.keep_cols)}
        LN({base_score} / (1 - {base_score})) as base_logit,
        {" + ".join(tree_sqls)} as tree_sum
    FROM {self.table_name}
),
margin AS (
    SELECT 
        {", ".join(self.keep_cols)},
        base_logit + tree_sum as margin_value
    FROM tree_scores
)
SELECT
    {", ".join(self.keep_cols)},
    CASE 
        WHEN 1 / (1 + EXP(-margin_value)) >= 0.5 THEN 1 
        ELSE 0 
    END AS prediction,
    1 / (1 + EXP(-margin_value)) AS probability
FROM margin
"""
        
        self.logger.info("XGBoost模型已转换为SQL语句")
        return {"type": "SQL", "model": final_sql}
