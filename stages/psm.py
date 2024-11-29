import json
import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Liquid, Grid

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
# import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from dags.stage import CustomStage
from stages.utils.plot_img import plot_proba_distribution


class PSM(CustomStage):
    
    def __init__(
        self, 
        cols: list[str], 
        need_normalize: bool = True,
        model_type: str = 'logistic',
        model_params: dict = None,
        *args, 
        **kwargs
    ):
        super().__init__(n_outputs=2)
        self.cols = cols
        self.need_normalize = need_normalize
        self.model_type = model_type
        
        # 默认模型参数
        default_params = {
            'logistic': {
                'random_state': 42,
                'max_iter': 1000
            },
            'rf': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 5
            },
            'gbdt': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'xgb': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'binary:logistic'
            },
            'lgb': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'binary'
            }
        }
        
        # 合并用户自定义参数
        self.model_params = {
            **default_params.get(model_type, {}),
            **(json.loads(model_params) if model_params else {})
        }
        
        # 初始化模型
        self.model = self._init_model()
        self.scaler = StandardScaler() if need_normalize else None
        
    def _init_model(self):
        """初始化选定的模型"""
        if self.model_type == 'logistic':
            return LogisticRegression(**self.model_params)
        elif self.model_type == 'rf':
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == 'gbdt':
            return GradientBoostingClassifier(**self.model_params)
        elif self.model_type == 'xgb':
            return xgb.XGBClassifier(**self.model_params)
        # elif self.model_type == 'lgb':
        #     return lgb.LGBMClassifier(**self.model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model_type == 'logistic':
            importance = np.abs(self.model.coef_[0])
        elif self.model_type in ['rf', 'gbdt', 'xgb', 'lgb']:
            importance = self.model.feature_importances_
        else:
            return None
            
        return dict(zip(self.cols, importance))

    def calculate_similarity(self, hist_A, hist_B, proba_A, proba_B, method):
        """计算两组概率分布的相似度"""
        if method == 'cosine':
            # 余弦相似度已经是标准化的，范围在[-1,1]之间
            return cosine_similarity(hist_A.reshape(1, -1), hist_B.reshape(1, -1))[0][0]
        elif method == 'manhattan':
            # Manhattan距离需要更好的归一化
            dist = manhattan_distances(hist_A.reshape(1, -1), hist_B.reshape(1, -1))[0][0]
            return np.exp(-dist)  # 使用指数转换，保证结果在(0,1]之间
        elif method == 'distribution_overlap':
            # 当前实现是正确的
            return 1 - np.mean(np.abs(hist_A - hist_B))
        elif method == 'jensen_shannon':
            # JS散度已经是对称的，范围在[0,1]之间
            return 1 - jensenshannon(hist_A, hist_B)
        elif method == 'wasserstein':
            # Wasserstein距离需要更好的归一化
            dist = wasserstein_distance(proba_A, proba_B)
            return np.exp(-dist)  # 使用指数转换
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")

    def forward(self, lz_A: pl.LazyFrame, lz_B: pl.LazyFrame):
        """执行PSM匹配"""
        # 检查是否有缺失值
        missing_A = lz_A.filter(pl.any_horizontal(pl.col(self.cols).is_null())).collect()
        missing_B = lz_B.filter(pl.any_horizontal(pl.col(self.cols).is_null())).collect()

        if not missing_A.is_empty() or not missing_B.is_empty():
            self.logger.error("输入数据存在缺失值，请注意处理")
            self.logger.error(f"实验组缺失值数据: {missing_A}")
            self.logger.error(f"对照组缺失值数据: {missing_B}")
            raise ValueError("输入数据存在缺失值，请注意处理")

        # 将lz_A和lz_B拼接在一起
        self.logger.info("将群体A设置为实验组")
        lz_A = lz_A.with_columns(pl.lit(1).alias("psm_label"))
        self.logger.info("将群体B设置为对照组")
        lz_B = lz_B.with_columns(pl.lit(0).alias("psm_label"))
        A_length = lz_A.select(pl.count()).collect().item()
        B_length = lz_B.select(pl.count()).collect().item()
        lz_train = pl.concat([lz_A, lz_B]).select(self.cols + ["psm_label"]).collect()

        if self.need_normalize:
            self.logger.warn("对特征进行归一化/标准化处理")
            lz_train[self.cols] = self.scaler.fit_transform(lz_train[self.cols])

        # 训练模型
        self.logger.info(f"训练模型, 特征: {self.cols}, 实验组样本数: {A_length}, 对照组样本数: {B_length}")
        self.model.fit(lz_train[self.cols], lz_train["psm_label"])
        
        # 获取预测概率
        if self.model_type in ['xgb', 'lgb']:
            proba = self.model.predict_proba(lz_train[self.cols])[:, 1]
        else:
            proba = self.model.predict_proba(lz_train[self.cols])[:, 1]
            
        # 获取特征重要性
        self.feature_importance = self.get_feature_importance()
        if self.feature_importance:
            importance_msg = "\n".join(
                f"- {col}: {imp:.4f}" 
                for col, imp in sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            )
            self.logger.info(f"特征重要性:\n{importance_msg}")

        # 获取两组的倾向性得分
        proba_A = proba[:A_length]
        proba_B = proba[A_length:]

        self.logger.info(f"实验组倾向得分范围: {proba_A.min()} ~ {proba_A.max()}")
        self.logger.info(f"对照组倾向得分范围: {proba_B.min()} ~ {proba_B.max()}")

        # 绘制概率分布对比图
        prob_dist_chart = plot_proba_distribution(proba_A, proba_B, n_bins=20, title="")
        self.summary.append({"倾向性得分分布": prob_dist_chart.dump_options_with_quotes()})

        return lz_A.with_columns(pl.lit(proba_A).alias("propensity_score")).lazy(), \
               lz_B.with_columns(pl.lit(proba_B).alias("propensity_score")).lazy()