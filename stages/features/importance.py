import logging
import polars as pl
import xgboost as xgb

from dags.stage import BaseStage


class XGBImportance(BaseStage):

    # 设置XGBoost参数
    base_params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.8,  # 随机采样比例
        'colsample_bytree': 0.8,  # 每棵树的列采样比例
        'gamma': 0.1,  # 最小损失减少
        'lambda': 1,  # L2正则化项
        'alpha': 0,  # L1正则化项
        'scale_pos_weight': 1  # 正负样本权重
    }

    def __init__(self, label_col: str, train_cols: list[str]=None, num_round=100, train_params: dict=base_params, topK: int=20,
                 importance_type: str="gain"):
        super().__init__(n_outputs=2)

        self.label_col = label_col
        self.train_cols = train_cols
        self.num_round = num_round
        self.train_params = train_params
        self.topK = topK
        self.importance_type = importance_type

    def forward(self, train_df: pl.DataFrame, transform_df: pl.DataFrame=None):

        if isinstance(train_df, pl.LazyFrame):
            train_df = train_df.collect()
        
        # 划分出数据集和label
        if self.train_cols is None:
            self.train_cols = [col for col in train_df.columns if col != self.label_col]

        if self.label_col in self.train_cols:
            logging.warn(f"检测到label在训练特征内，自动剔除")
            self.train_cols.remove(self.label_col)

        logging.info(f"训练数据集特征: {self.train_cols}")
        logging.info(f"训练数据集label: {self.label_col}")
        logging.info(f"训练参数: {self.train_params}")

        X = train_df.select(self.train_cols)
        y = train_df.select(self.label_col)

        # 将数据转换为XGBoost可用的DMatrix格式
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.train_cols)

        model = xgb.train(self.train_params, dtrain, self.num_round)

        # 特征重要性
        importance = model.get_score(importance_type=self.importance_type)
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        out_features = []

        logging.info(f"特征重要性排名前{self.topK}：")
        for feature, score in importance[:self.topK]:
            logging.info(f"{feature}: {score}")
            out_features.append(feature)

        if transform_df is not None:
            return out_features, transform_df.lazy().select(out_features)
        else:
            return out_features, None

        



