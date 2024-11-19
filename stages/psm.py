import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Line, Liquid, Page, Grid
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from dags.stage import CustomStage


class PSM(CustomStage):
    
    def __init__(self, cols: list[str], need_normalize: bool = True, similarity_method: str = 'cosine', 
                 model_params: dict = {}, *args, **kwargs):
        super().__init__(n_outputs=0)
        self.cols = cols                                        # 特征列    
        self.need_normalize = need_normalize                    # 是否需要归一化/标准化
        self.similarity_method = similarity_method              # 相似度计算方法

        # 模型参数
        self.model_params = model_params if model_params else {
            'random_state': 42,
            'max_iter': 1000
        }

        self.scaler = StandardScaler()
        self.model = LogisticRegression(**self.model_params)

    def calculate_similarity(self, proba_A, proba_B, method):
        """计算两组概率分布的相似度"""
        
        # 确保输入是一维数组
        proba_A = np.array(proba_A).ravel()
        proba_B = np.array(proba_B).ravel()
        
        # 计算直方图数据
        # 自动确定最优bins数量
        edges = np.histogram_bin_edges(
            np.concatenate([proba_A, proba_B]),
            bins='auto'  # 或者使用 'sturges', 'fd' 等方法
        )
        
        # 使用相同的bins边界计算直方图
        hist_A, _ = np.histogram(proba_A, bins=edges, density=True)
        hist_B, _ = np.histogram(proba_B, bins=edges, density=True)
        
        # 将直方图数据标准化
        hist_A = hist_A / np.sum(hist_A)
        hist_B = hist_B / np.sum(hist_B)
        
        if method == 'cosine':
            return cosine_similarity(hist_A.reshape(1, -1), hist_B.reshape(1, -1))[0][0]
        elif method == 'manhattan':
            return 1 / (1 + manhattan_distances(hist_A.reshape(1, -1), hist_B.reshape(1, -1))[0][0])
        elif method == 'distribution_overlap':
            return 1 - np.mean(np.abs(hist_A - hist_B))
        elif method == 'jensen_shannon':
            return 1 - jensenshannon(hist_A, hist_B)
        elif method == 'wasserstein':
            # 对于Wasserstein距离，我们使用原始数据而不是直方图
            return 1 / (1 + wasserstein_distance(proba_A, proba_B))
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")

    def plot_distribution(self, proba_A, proba_B):
        """使用pyecharts绘制概率分布对比图"""
        # 计算直方图数据
        # 自动确定最优bins数量
        edges = np.histogram_bin_edges(
            np.concatenate([proba_A, proba_B]),
            bins='auto'  # 或者使用 'sturges', 'fd' 等方法
        )
        
        # 使用相同的bins边界计算直方图
        hist_A, _ = np.histogram(proba_A, bins=edges, density=True)
        hist_B, _ = np.histogram(proba_B, bins=edges, density=True)
        
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
            series_name="群体A[种子用户]",
            y_axis=hist_A.tolist(),
            symbol_size=8,
            is_smooth=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2)
        )
        
        # 添加相似用户分布曲线
        line.add_yaxis(
            series_name="群体B[相似用户]",
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
        """
        进行倾向性得分匹配
        A 处理组 label=1
        B 对照组 label=0
        """

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
        proba = self.model.predict_proba(lz_train[self.cols])[:, 1]
        proba_A = proba[:A_length]
        proba_B = proba[A_length:]

        # AUC 接近0.5 说明两组人群较为相似
        auc = roc_auc_score(lz_train["psm_label"], proba)
        # KS统计量越小，表示两组分布越接近
        fpr, tpr, _ = roc_curve(lz_train["psm_label"], proba)
        ks = max(abs(tpr - fpr))
        # 重叠度 越接近1，表示两组分布越接近
        overlap = 1 - ks
        # 相似度计算
        similarity = self.calculate_similarity(proba_A, proba_B, self.similarity_method)

        # 在计算完所有指标后，添加可视化
        metrics = {
            'auc': auc,
            'ks': ks,
            'overlap': overlap,
            'similarity': similarity
        }
        
        # 创建分布对比图和指标水滴图
        distribution_chart = self.plot_distribution(proba_A, proba_B)
        metrics_chart = self.plot_metrics(metrics)

        self.summary = [{
            '模型指标': metrics_chart.dump_options_with_quotes(),
            '概率分布对比': distribution_chart.dump_options_with_quotes()
        }]
        
        # 保存图表
        # distribution_chart.render(f"./psm_distribution.html")
        # metrics_chart.render(f"./psm_metrics.html")
        
        # 记录评估指标
        self.logger.warn(f"AUC接近0.5 说明两组人群较为相似")
        self.logger.warn(f"KS越小，表示两组分布越接近")
        self.logger.warn(f"PSM评估指标：AUC={auc:.4f}, KS={ks:.4f}, 重叠度={overlap:.4f}, 相似度={similarity:.4f}")

        # TODO 用户匹配 & 计算匹配度
        
        
        
