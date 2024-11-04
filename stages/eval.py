import polars as pl
from pyecharts.charts import Line, HeatMap, Gauge, Liquid, Grid
from pyecharts import options as opts
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pyecharts import options as opts
from pyecharts.charts import Grid, Liquid
from pyecharts.commons.utils import JsCode

from dags.stage import CustomStage


class BinaryEval(CustomStage):

    def __init__(self, y):
        super().__init__(n_outputs=1)
        self.y = y

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

        # ROC曲线
        roc_line = (
            Line()
            .add_xaxis([float(f) for f in fpr])
            .add_yaxis("ROC曲线", [float(t) for t in tpr])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"ROC曲线 (AUC={roc_auc:.3f})"),
                xaxis_opts=opts.AxisOpts(name="假阳性率"),
                yaxis_opts=opts.AxisOpts(name="真阳性率")
            )
        )

        # PR曲线 
        pr_line = (
            Line()
            .add_xaxis([float(r) for r in recall])
            .add_yaxis("PR曲线", [float(p) for p in precision])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"PR曲线 (AUC={pr_auc:.3f})"),
                xaxis_opts=opts.AxisOpts(name="召回率"),
                yaxis_opts=opts.AxisOpts(name="精确率")
            )
        )

        # 阈值-召回率曲线
        recall_line = (
            Line()
            .add_xaxis([float(t) for t in thresholds])
            .add_yaxis("召回率", [float(r) for r in tpr])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="阈值-召回率曲线"),
                xaxis_opts=opts.AxisOpts(name="阈值"),
                yaxis_opts=opts.AxisOpts(name="召回率")
            )
        )

        # 计算混淆矩阵
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()

        # 混淆矩阵热力图
        confusion_heatmap = (
            HeatMap()
            .add_xaxis(["预测负例", "预测正例"])
            .add_yaxis(
                "实际类别",
                ["实际负例", "实际正例"],
                [[0, 0, tn], [0, 1, fp], [1, 0, fn], [1, 1, tp]]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="混淆矩阵"),
                visualmap_opts=opts.VisualMapOpts(min_=0),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(type_="category")
            )
        )

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

        # 准确率
        accuracy_liquid = (
            Liquid()
                .add("准确率", [accuracy], center=["20%", "50%"], 
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
                        pos_left="20%",  
                        pos_bottom="5%",
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
                .add("精确率", [precision_score], center=["40%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        # formatter=JsCode(
                        #     """function (param) {
                        #             return (Math.floor(param.value * 10000) / 100) + '%';
                        #         }"""
                        # ),
                        position="inside",
                    ),
                    color=["#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC"],
                    background_color="#F3E5F5",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#F3E5F5", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="精确率",
                        pos_left="40%",  
                        pos_bottom="5%",
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
                .add("召回率", [recall_score], center=["60%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        # formatter=JsCode(
                        #     """function (param) {
                        #             return (Math.floor(param.value * 10000) / 100) + '%';
                        #         }"""
                        # ),
                        position="inside",
                    ),
                    color=["#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A"],
                    background_color="#E8F5E9",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#E8F5E9", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="召回率",
                        pos_left="60%",  
                        pos_bottom="5%",
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
                .add("F1分数", [f1], center=["80%", "50%"],
                    label_opts=opts.LabelOpts(
                        font_size=20,
                        # formatter=JsCode(
                        #     """function (param) {
                        #             return (Math.floor(param.value * 10000) / 100) + '%';
                        #         }"""
                        # ),
                        position="inside",
                    ),
                    color=["#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726"],
                    background_color="#FFF3E0",
                    outline_itemstyle_opts=opts.ItemStyleOpts(border_color="#FFF3E0", border_width=3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="F1分数",
                        pos_left="80%",  
                        pos_bottom="5%", 
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=14,
                            color="#000"
                        )
                    )
                )
        )

        # 使用 Grid 类将四个图表布局在一个画布上
        base_metrics = (
            Grid(init_opts=opts.InitOpts())
                .add(accuracy_liquid, grid_opts=opts.GridOpts(
                    pos_left="5%",     # 左边留5%边距
                    pos_right="77%"    # 右侧留出位置给第二个图表
                ))
                .add(precision_liquid, grid_opts=opts.GridOpts(
                    pos_left="28%",    # 从28%位置开始
                    pos_right="54%"    # 留出位置给第三个图表
                ))
                .add(recall_liquid, grid_opts=opts.GridOpts(
                    pos_left="51%",    # 从51%位置开始
                    pos_right="31%"    # 留出位置给第四个图表
                ))
                .add(f1_liquid, grid_opts=opts.GridOpts(
                    pos_left="74%",    # 从74%位置开始
                    pos_right="5%"     # 右边留5%边距
                ))
        )

        # 渲染图表
        base_metrics.render("base_metrics.html")

        metrics = {
            "accuracy": accuracy,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
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
            {"roc": roc_line.dump_options_with_quotes()},
            {"pr": pr_line.dump_options_with_quotes()},
            {"recall": recall_line.dump_options_with_quotes()},
            {"confusion": confusion_heatmap.dump_options_with_quotes()},
        ])

        self.logger.info(f"评估指标: {metrics}")
        return lf