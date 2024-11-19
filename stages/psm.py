import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Line, Liquid, Grid
from pyecharts.globals import ThemeType

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
# import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy import stats
from dags.stage import CustomStage
from stages.distance_match import DistanceMatch


class PSM(CustomStage):
    
    def __init__(
        self, 
        cols: list[str], 
        need_normalize: bool = True,
        similarity_method: str = 'cosine',
        model_type: str = 'logistic',
        model_params: dict = None,
        *args, 
        **kwargs
    ):
        super().__init__(n_outputs=2)
        self.cols = cols
        self.need_normalize = need_normalize
        self.similarity_method = similarity_method
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
            **(model_params or {})
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

    def plot_distribution(self, hist_A, hist_B, edges):
        """使用pyecharts绘制概率分布对比图"""
        
        # 计算中点作为x轴
        x_A = (edges[:-1] + edges[1:]) / 2
        
        line = Line(
            init_opts=opts.InitOpts(
                theme=ThemeType.LIGHT,
                width="900px",
                height="500px"
            )
        )
        
        line.add_xaxis(xaxis_data=[f"{x:.3f}" for x in x_A])
        
        # 添加种子用户分布曲线
        line.add_yaxis(
            series_name="实验组",
            y_axis=hist_A.tolist(),
            symbol_size=8,
            is_smooth=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
        
        # 添加相似用户分布曲线
        line.add_yaxis(
            series_name="对照组",
            y_axis=hist_B.tolist(),
            symbol_size=8,
            is_smooth=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
        
        # 设置全局选项
        line.set_global_opts(
            title_opts=opts.TitleOpts(
                title="PSM概率分布对比",
                pos_left="center"
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            xaxis_opts=opts.AxisOpts(
                name="Propensity Score",
                name_location="center",
                name_gap=35,
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                name="密度",
                name_location="center",
                name_gap=40,
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ],
        )
        
        return line

    def plot_metrics(self, metrics):
        """使用Grid布局绘制指标水滴图"""
        # 创建一个Grid
        grid = Grid(
            # init_opts=opts.InitOpts(
            #     width="1200px",
            #     height="300px"  # 减小高度使图表更紧凑
            # )
        )
        
        # 创建四个水滴图
        metrics_charts = [
            (metrics['auc'], "AUC Score", "16.67%"),
            (metrics['ks'], "KS Statistic", "38.33%"),
            (metrics['overlap'], "Distribution Overlap", "61.67%"),
            (metrics['similarity'], "Similarity Score", "83.33%")
        ]
        
        for value, title, pos_left in metrics_charts:
            liquid = (
                Liquid()
                .add(
                    series_name=title,
                    data=[value],
                    center=[pos_left, "50%"],  # 使用center参数定位
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                        # formatter=JsCode(
                        #     """function (param) {
                        #         return Math.round(param.value * 100) + '%';
                        #     }"""
                        # ),
                    ),
                    color=["#294D99", "#156ACF", "#1598ED", "#45BDFF"],
                    background_color="#E1F5FE",
                    outline_itemstyle_opts=opts.ItemStyleOpts(
                        border_color="#E1F5FE",
                        border_width=3
                    )
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=title,
                        pos_left=pos_left,
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
            )
            
            # 添加到Grid中，使用相对位置
            grid.add(
                liquid,
                grid_opts=opts.GridOpts(
                    pos_left=pos_left,
                    pos_right=f"{100 - float(pos_left.strip('%')) - 16.67}%",
                    pos_top="15%",
                    pos_bottom="15%"
                )
            )
        
        return grid

    def forward(self, lz_A: pl.LazyFrame, lz_B: pl.LazyFrame):
        """执行PSM匹配"""
        # 检查是否有缺失值
        missing_A = lz_A.filter(pl.any_horizontal(pl.col(self.cols).is_null())).collect()
        missing_B = lz_B.filter(pl.any_horizontal(pl.col(self.cols).is_null())).collect()

        if not missing_A.is_empty() or not missing_B.is_empty():
            self.logger.error("输入数据存在缺失值，请注意处理")
            raise ValueError("输入数据存在缺失值，请注意处理")

        # 将lz_A和lz_B拼接在一起
        lz_A = lz_A.with_columns(pl.lit(1).alias("psm_label"))
        lz_B = lz_B.with_columns(pl.lit(0).alias("psm_label"))
        A_length = lz_A.select(pl.count()).collect().item()
        lz_train = pl.concat([lz_A, lz_B]).select(self.cols + ["psm_label"]).collect()

        if self.need_normalize:
            self.logger.warn("对特征进行归一化/标准化处理")
            lz_train[self.cols] = self.scaler.fit_transform(lz_train[self.cols])

        # 训练模型
        self.model.fit(lz_train[self.cols], lz_train["psm_label"])
        
        # 获取预测概率
        if self.model_type in ['xgb', 'lgb']:
            proba = self.model.predict_proba(lz_train[self.cols])[:, 1]
        else:
            proba = self.model.predict_proba(lz_train[self.cols])[:, 1]
            
        # 获取特征重要性
        feature_importance = self.get_feature_importance()
        if feature_importance:
            importance_msg = "\n".join(
                f"- {col}: {imp:.4f}" 
                for col, imp in sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            )
            self.logger.info(f"特征重要性:\n{importance_msg}")
            
        # AUC 接近0.5 说明两组人群较为相似
        auc = roc_auc_score(lz_train["psm_label"], proba)

        # 获取两组的倾向性得分
        proba_A = proba[:A_length]
        proba_B = proba[A_length:]
        
        # 计算KS统计量
        def calculate_ks(scores_A, scores_B):
            """计算标准化后的KS统计量
            
            Args:
                scores_A: 处理组的倾向性得分
                scores_B: 对照组的倾向性得分
                
            Returns:
                float: 标准化的KS统计量 [0,1]
            """
            # 计算经验分布函数
            def empirical_cdf(x, sample):
                return np.sum(sample <= x) / len(sample)
            
            # 获取所有唯一的得分点
            all_scores = np.sort(np.unique(np.concatenate([scores_A, scores_B])))
            
            # 计算两个分布的CDF差异
            cdf_diffs = np.array([
                abs(empirical_cdf(x, scores_A) - empirical_cdf(x, scores_B))
                for x in all_scores
            ])
            
            # 返回最大差异(KS统计量)
            return np.max(cdf_diffs)
        
        # 计算KS统计量
        ks = calculate_ks(proba_A, proba_B)
        
        # 评估KS值
        ks_quality = (
            "非常相似" if ks < 0.1 else
            "比较相似" if ks < 0.2 else
            "差异一般" if ks < 0.3 else
            "差异较大"
        )
        
        self.logger.warn(f"KS统计量解释：")
        self.logger.warn(f"- KS={ks:.4f} ({ks_quality})")
        self.logger.warn(f"- KS < 0.1: 两组分布非常相似")
        self.logger.warn(f"- KS < 0.2: 两组分布比较相似")
        self.logger.warn(f"- KS < 0.3: 两组分布差异一般")
        self.logger.warn(f"- KS >= 0.3: 两组分布差异较大")

        # 计算直方图时建议使用更稳健的方法
        min_score = min(proba[:A_length].min(), proba[A_length:].min())
        max_score = max(proba[:A_length].max(), proba[A_length:].max())
        
        # 使用Freedman-Diaconis规则确定bin宽度
        def get_bin_width(x):
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            return 2 * iqr / (len(x) ** (1/3))
        
        bin_width = min(get_bin_width(proba[:A_length]), get_bin_width(proba[A_length:]))
        n_bins = int(np.ceil((max_score - min_score) / bin_width))
        bins = np.linspace(min_score, max_score, n_bins)
        
        # 计算并标准化直方图
        hist_A, _ = np.histogram(proba[:A_length], bins=bins, density=True)
        hist_B, _ = np.histogram(proba[A_length:], bins=bins, density=True)

        # 重叠度计算
        overlap = np.minimum(hist_A, hist_B).sum() / np.maximum(hist_A, hist_B).sum()

        # 相似度计算
        similarity = self.calculate_similarity(hist_A, hist_B, proba[:A_length], proba[A_length:], self.similarity_method)

        # 在计算完所有指标后，添加可视化
        metrics = {
            'auc': auc,
            'ks': ks,
            'overlap': overlap,
            'similarity': similarity
        }
        
        # 创建分布对比图和指标水滴图
        distribution_chart = self.plot_distribution(hist_A, hist_B, bins)
        metrics_chart = self.plot_metrics(metrics)

        self.summary = [{
            '模型指标': metrics_chart.dump_options_with_quotes(),
            '概率分布对比': distribution_chart.dump_options_with_quotes()
        }]
        
        # 添加更详细的评估指标解释
        auc_quality = "差异较大" if abs(auc - 0.5) > 0.1 else "相似"
        ks_quality = "差异较大" if ks > 0.1 else "相似"
        overlap_quality = "相似度高" if overlap > 0.8 else "相似度一般" if overlap > 0.6 else "差异较大"
        
        self.logger.warn(f"PSM评估指标解释：")
        self.logger.warn(f"- AUC={auc:.4f} ({auc_quality})")
        self.logger.warn(f"- 分布重叠度={overlap:.4f} ({overlap_quality})")
        self.logger.warn(f"- {self.similarity_method}相似度={similarity:.4f}")

        return lz_A.with_columns(pl.lit(proba_A).alias("psm_score")).lazy(), lz_B.with_columns(pl.lit(proba_B).alias("psm_score")).lazy()
