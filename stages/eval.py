import polars as pl
from pyecharts.charts import Line, HeatMap, Gauge, Liquid, Grid
from pyecharts import options as opts
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pyecharts import options as opts
from pyecharts.charts import Grid, Liquid
from pyecharts.commons.utils import JsCode
from scipy import stats
import numpy as np

from dags.stage import CustomStage


class BinaryEval(CustomStage):

    def __init__(self, y):
        super().__init__(n_outputs=1)
        self.y = y

    def downsample_curve_data(self, x_data, y_data, n_points=100):
        """对曲线数据进行降采样
        
        Args:
            x_data: x轴数据
            y_data: y轴数据 
            n_points: 目标采样点数
        """
        if len(x_data) <= n_points:
            return x_data, y_data
            
        # 计算采样间隔
        step = len(x_data) // n_points
        
        # 降采样
        x_sampled = x_data[::step]
        y_sampled = y_data[::step]
        
        # 确保包含首尾点
        if x_data[-1] not in x_sampled:
            x_sampled = np.append(x_sampled, x_data[-1])
            y_sampled = np.append(y_sampled, y_data[-1])
            
        return x_sampled, y_sampled

    def forward(self, lf: pl.LazyFrame):
        
        # 收集数据
        df = lf.select([pl.col(self.y), pl.col("y_score"), pl.col("y_pred")]).collect().to_pandas()
        y_true = df[self.y].to_numpy()
        y_score = df["y_score"].to_numpy() 
        y_pred = df["y_pred"].to_numpy()

        # 计算各项指标
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        # 计算KS值
        ks = max(abs(tpr - fpr))

        # ROC曲线
        fpr_sampled, tpr_sampled = self.downsample_curve_data(fpr, tpr)
        roc_line = (
            Line()
            .add_xaxis(
                [round(float(f), 2) for f in fpr_sampled]
            )
            .add_yaxis(
                "ROC曲线", 
                [round(float(t), 2) for t in tpr_sampled],
                symbol="circle",
                symbol_size=4,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(
                    width=3  # 设置线条粗细
                ),
                label_opts=opts.LabelOpts(
                    is_show=False  # 默认不显示标签
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",  # 坐标轴触发
                    axis_pointer_type="cross",  # 十字准星指示器
                    formatter="""
                        假阳性率: {0}<br/>
                        真阳性率: {1}
                    """.format("{c0}", "{c1}")
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"ROC曲线 (AUC={roc_auc:.3f})"),
                xaxis_opts=opts.AxisOpts(
                    name="假阳性率",
                    type_="value",
                    min_=0,
                    max_=1
                ),
                yaxis_opts=opts.AxisOpts(
                    name="真阳性率",
                    type_="value",
                    min_=0,
                    max_=1
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    formatter="{c}"
                ),
            )
        )
        # 使用 Grid 调整布局
        roc_grid = Grid()
        roc_grid.add(
            roc_line,
            grid_opts=opts.GridOpts(
                pos_left="15%",
                pos_right="15%",
                pos_top="15%",
                pos_bottom="15%"
            )
        )

        # PR曲线 
        recall_sampled, precision_sampled = self.downsample_curve_data(recall, precision)
        pr_line = (
            Line()
            .add_xaxis(
                [round(float(r), 4) for r in recall_sampled]
            )
            .add_yaxis(
                "PR曲线", 
                [round(float(p), 4) for p in precision_sampled],
                symbol="circle",
                symbol_size=4,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(
                    width=3
                ),
                label_opts=opts.LabelOpts(
                    is_show=False
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"PR曲线 (AUC={pr_auc:.3f})"),
                xaxis_opts=opts.AxisOpts(
                    name="召回率",
                    type_="value",
                    min_=0,
                    max_=1
                ),
                yaxis_opts=opts.AxisOpts(
                    name="精确率",
                    type_="value",
                    min_=0,
                    max_=1
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",  # 坐标轴触发
                    axis_pointer_type="cross",  # 十字准星指示器
                )
            )
        )

        # 使用 Grid 调整 PR 曲线布局
        pr_grid = Grid()
        pr_grid.add(
            pr_line,
            grid_opts=opts.GridOpts(
                pos_left="15%",
                pos_right="15%",
                pos_top="15%",
                pos_bottom="15%"
            )
        )

        # 阈值-召回率曲线
        thresholds_sampled, tpr_sampled = self.downsample_curve_data(thresholds, tpr)
        recall_line = (
            Line()
            .add_xaxis(
                [round(float(t), 4) for t in thresholds_sampled]
            )
            .add_yaxis(
                "召回率", 
                [round(float(r), 4) for r in tpr_sampled],
                symbol="circle",
                symbol_size=4,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(
                    width=3
                ),
                label_opts=opts.LabelOpts(
                    is_show=False
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="阈值-召回率曲线"),
                xaxis_opts=opts.AxisOpts(
                    name="阈值",
                    type_="value"
                ),
                yaxis_opts=opts.AxisOpts(
                    name="召回率",
                    type_="value",
                    min_=0,
                    max_=1
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",  # 坐标轴触发
                    axis_pointer_type="cross",  # 十字准星指示器
                )
            )
        )

        # 使用 Grid 调整阈值-召回率曲线布局
        recall_grid = Grid()
        recall_grid.add(
            recall_line,
            grid_opts=opts.GridOpts(
                pos_left="15%",
                pos_right="15%",
                pos_top="15%",
                pos_bottom="15%"
            )
        )

        # 计算混淆矩阵
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()

        # 准备热力图数据
        value = [
            [0, 0, int(tn)],  # [x, y, value]
            [1, 0, int(fp)],
            [0, 1, int(fn)],
            [1, 1, int(tp)]
        ]

        # 混淆矩阵热力图
        confusion_heatmap = (
            HeatMap()
            .add_xaxis(["预测负例", "预测正例"])
            .add_yaxis(
                "实际类别",
                ["实际负例", "实际正例"],
                value,
                label_opts=opts.LabelOpts(
                    position="middle",     # 在格子中间显示数值
                    font_size=16,         # 设置字体大小
                    font_weight="bold"    # 设置字体粗细
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="混淆矩阵"),
                visualmap_opts=opts.VisualMapOpts(),
            )
        )

        # 使用 Grid 调整混淆矩阵布局
        confusion_grid = Grid()
        confusion_grid.add(
            confusion_heatmap,
            grid_opts=opts.GridOpts(
                pos_left="15%",
                pos_right="20%",  # 留出右边的空间给颜色条
                pos_top="15%",
                pos_bottom="15%"
            )
        )

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

        # 准确率
        accuracy_liquid = (
            Liquid()
                .add("准确率", [accuracy], center=["16.67%", "50%"], 
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                    ),
                    color=["#B3E5FC", "#81D4FA", "#4FC3F7", "#29B6F6"],
                    background_color="#E1F5FE",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#E1F5FE", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="准确率",
                        pos_left="16.67%",  
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # 精确率
        precision_liquid = (
            Liquid()
                .add("精确率", [precision_score], center=["33.33%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                    ),
                    color=["#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC"],
                    background_color="#F3E5F5",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#F3E5F5", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="精确率",
                        pos_left="33.33%",  
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # 召回率
        recall_liquid = (
            Liquid()
                .add("召回率", [recall_score], center=["50%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                    ),
                    color=["#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A"],
                    background_color="#E8F5E9",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#E8F5E9", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="召回率",
                        pos_left="50%",  
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # F1分数
        f1_liquid = (
            Liquid()
                .add("F1分数", [f1], center=["66.67%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                    ),
                    color=["#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726"],
                    background_color="#FFF3E0",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#FFF3E0", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="F1分数",
                        pos_left="66.67%",  
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # KS值
        ks_liquid = (
            Liquid()
                .add("KS值", [ks], center=["83.33%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        position="inside",
                    ),
                    color=["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5"],
                    background_color="#E3F2FD",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#E3F2FD", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="KS值",
                        pos_left="83.33%",  
                        pos_bottom="10%",
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # 使用 Grid 类将五个图表布局在一个画布上
        base_metrics = (
            Grid(init_opts=opts.InitOpts())
                .add(accuracy_liquid, grid_opts=opts.GridOpts(
                    pos_left="5%",     
                    pos_right="81%"    
                ))
                .add(precision_liquid, grid_opts=opts.GridOpts(
                    pos_left="21.67%",    
                    pos_right="64.33%"    
                ))
                .add(recall_liquid, grid_opts=opts.GridOpts(
                    pos_left="38.33%",    
                    pos_right="47.67%"    
                ))
                .add(f1_liquid, grid_opts=opts.GridOpts(
                    pos_left="55%",    
                    pos_right="31%"    
                ))
                .add(ks_liquid, grid_opts=opts.GridOpts(
                    pos_left="71.67%",    
                    pos_right="14.33%"    
                ))
        )

        # 计算核密度估计
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        
        # 计算正样本的核密度估计
        pos_density = stats.gaussian_kde(pos_scores)
        x_range = np.linspace(min(y_score), max(y_score), 200)
        pos_y = pos_density(x_range)
        
        # 计算负样本的核密度估计
        neg_density = stats.gaussian_kde(neg_scores)
        neg_y = neg_density(x_range)

        # 创建双峰图
        density_line = (
            Line()
            .add_xaxis([round(float(x), 4) for x in x_range])
            .add_yaxis(
                "正样本[y_true==1]",
                [round(float(y), 4) for y in pos_y],
                symbol="none",
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(width=3),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#66BB6A"),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3)
            )
            .add_yaxis(
                "负样本[y_true==0]",
                [round(float(y), 4) for y in neg_y],
                symbol="none",
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(width=3),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#EF5350"),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="预测分数分布"),
                xaxis_opts=opts.AxisOpts(
                    name="预测分数",
                    type_="value",
                ),
                yaxis_opts=opts.AxisOpts(
                    name="密度",
                    type_="value",
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                )
            )
        )
        
        # 使用Grid调整布局
        density_grid = Grid()
        density_grid.add(
            density_line,
            grid_opts=opts.GridOpts(
                pos_left="15%",
                pos_right="15%",
                pos_top="15%",
                pos_bottom="15%"
            )
        )

        # 渲染图表
        # base_metrics.render("base_metrics.html")

        metrics = {
            "accuracy": accuracy,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "ks": ks,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp), 
                "fn": int(fn),
                "tp": int(tp)
            },
        }
        self.logger.info(metrics)

        self.summary.extend([
            {"基础评估指标": base_metrics.dump_options_with_quotes()},
            {"ROC曲线": roc_grid.dump_options_with_quotes()},
            {"PR曲线": pr_grid.dump_options_with_quotes()},
            {"Recall曲线": recall_grid.dump_options_with_quotes()},
            {"双峰图": density_grid.dump_options_with_quotes()},
            {"混淆矩阵": confusion_grid.dump_options_with_quotes()},
        ])

        self.logger.info(f"评估指标: {metrics}")
        return lf
    

