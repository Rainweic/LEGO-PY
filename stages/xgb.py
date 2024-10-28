import polars as pl
import xgboost as xgb

from pyecharts.charts import Bar
from pyecharts import options as opts
from dags.stage import BaseStage


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

class XGB(BaseStage):

    def __init__(self, label_col: str, train_cols: list[str]=None, num_round=100, train_params: dict=base_params):
        super().__init__(n_outputs=1)

        self.label_col = label_col
        self.train_cols = train_cols
        self.num_round = num_round
        if isinstance(train_params, str):
            import json
            train_params = json.loads(train_params)
        self.train_params = train_params

    def train(self, train_df: pl.DataFrame, eval_df: pl.DataFrame = None):

        if isinstance(train_df, pl.LazyFrame):
            train_df = train_df.collect()
        
        # 划分出数据集和label
        if self.train_cols is None:
            self.train_cols = [col for col in train_df.columns if col != self.label_col]

        if self.label_col in self.train_cols:
            self.logger.warn(f"检测到label在训练特征内，自动剔除")
            self.train_cols.remove(self.label_col)

        self.logger.info(f"训练数据集特征: {self.train_cols}")
        self.logger.info(f"训练数据集label: {self.label_col}")
        self.logger.info(f"训练参数: {self.train_params}")

        X = train_df.select(self.train_cols)
        y = train_df.select(self.label_col)

        # 将数据转换为XGBoost可用的DMatrix格式
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.train_cols)

        evals_result = {}
        model = xgb.train(self.train_params, dtrain, self.num_round, evals_result=evals_result)

        self.logger.info(f"训练集评估结果: {evals_result}")
        
        if eval_df is not None:
            # 评估
            if isinstance(eval_df, pl.LazyFrame):
                eval_df = eval_df.collect()

            X_eval = eval_df.select(self.train_cols)
            y_eval = eval_df.select(self.label_col)
            deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=self.train_cols)
            deval_evals_result = model.eval(deval)
            self.logger.info(f"验证集评估结果: {deval_evals_result}")
        
        return {"cols": self.train_cols, "xgb": model}
    
    @staticmethod
    def predict(model, lf: pl.LazyFrame):
        cols = model["cols"]
        model = model["xgb"]
        if isinstance(lf, pl.LazyFrame):
            X_test = lf.select(cols).collect()
        else:
            X_test = lf.select(cols)
        X_test = xgb.DMatrix(X_test, feature_names=cols)
        y_score = model.predict(X_test)
        return lf.with_columns(pl.lit(y_score).alias("y_score"))
    
    def forward(self, train_df: pl.DataFrame, eval_df: pl.DataFrame=None):
        return self.train(train_df=train_df, eval_df=eval_df)


class XGBImportance(XGB):

    def __init__(self, label_col: str=None, train_cols: list[str]=None, num_round=100, train_params: dict=base_params, topK: int=20,
                 importance_type: str="gain"):
        super().__init__(label_col=label_col, train_cols=train_cols, num_round=num_round, train_params=train_params)
        self._n_outputs = 2
        self.topK = topK
        self.importance_type = importance_type

    def forward(self, train_df: pl.DataFrame, model_xgb_f_importance: list[str]=None):

        if model_xgb_f_importance:
            # 上一个模型复用
            return train_df.lazy().select(model_xgb_f_importance), model_xgb_f_importance

        model = self.train(train_df, None)

        # 绘图
        f_i_bar_list = []
        for imp_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            importance = model.get_score(importance_type=imp_type)
            cols = list(importance.keys())
            data = [importance[c] for c in cols]
            f_i_bar = (
                Bar()
                .add_xaxis(cols)
                .add_yaxis(imp_type, data)
                .set_global_opts(title_opts=opts.TitleOpts(title=f"特征重要性-{imp_type}"))
            )
            f_i_bar_list.append({imp_type: f_i_bar.dump_options_with_quotes()})
        
        # 写入summary
        self.summary.extend(f_i_bar_list)
        # print(self.summary)

        # 获取特征重要性
        importance = model.get_score(importance_type=self.importance_type)
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        out_features = []
        self.logger.info(f"特征重要性排名前{self.topK}：")
        for feature, score in importance[:self.topK]:
            self.logger.info(f"{feature}: {score}")
            out_features.append(feature)

        return train_df.lazy().select(out_features), out_features

        



