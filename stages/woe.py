import polars as pl
import numpy as np
from typing import Dict, List, Optional
from dags.stage import CustomStage
from stages.utils.binning import Binning
from pyecharts.commons.utils import JsCode


class WOE(CustomStage):
    """WOE(Weight of Evidence)编码转换器
    
    将连续型或分类型特征转换为WOE编码。支持多种分箱方法，包括等频、等宽、卡方等。
    WOE = ln(好样本占比/坏样本占比)，可以将特征转换为与目标变量的相关性度量。

    参数:
        cols (Union[str, List[str]]): 
            需要进行WOE编码的列名，可以是单个字符串或字符串列表
        
        target_col (str): 
            目标变量的列名，必须是二值型变量(0/1)
        
        binning_method (str, 默认='quantile'): 
            分箱方法，支持以下几种:
            - 'equal_width': 等宽分箱
            - 'equal_freq': 等频分箱(quantile)
            - 'chi2': 卡方分箱(基于目标变量)
            - 'kmeans': 基于聚类的分箱
            - 'mdlp': 最小描述长度分箱
        
        n_bins (int, 默认=10): 
            分箱数量，仅在使用等、等频、kmeans分箱时有效
        
        bins (Optional[Dict[str, List[float]]], 默认=None): 
            自定义分箱点，格式为 {列名: [分箱点列表]}
            如果某列指定了自定义分箱点，将忽略binning_method和n_bins参数
        
        min_samples (float, 默认=0.05): 
            每个分箱最小样本比例，用于防止过度分箱
        
        max_bins (int, 默认=50): 
            最大分箱数量，用于防止过度分箱
        
        chi_merge_threshold (float, 默认=0.1): 
            卡方分箱时的合并阈值，较大的值会产生更少的分箱
        
        save_ori_col (bool, 默认=True):
            是否在结果中保留原始特征列
    
    返回:
        对于每个输入特征col，会生成两个新列：
        - {col}_bin: 分箱编号（从0开始的整数）
        - {col}_woe: WOE编码值

    示例:
        ```python
        # 基本使用
        woe = WOE(
            cols=['age', 'income'],
            target_col='is_default',
            binning_method='equal_freq',
            n_bins=10
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌──────┬──────────┬──────────┐
        │ age  ┆ age_bin  ┆ age_woe  │
        │ ---  ┆ ---      ┆ ---      │
        │ f64  ┆ i64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ 25.0 ┆ 2        ┆ 0.123    │
        │ 35.0 ┆ 3        ┆ -0.456   │
        └──────┴──────────┴──────────┘

        # 使用自定义分箱点
        woe = WOE(
            cols=['age'],
            target_col='is_default',
            bins={'age': [0, 18, 25, 35, 50, 65, float('inf')]}
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌──────┬──────────┬──────────┐
        │ age  ┆ age_bin  ┆ age_woe  │
        │ ---  ┆ ---      ┆ ---      │
        │ f64  ┆ i64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ 25.0 ┆ 2        ┆ 0.123    │
        │ 35.0 ┆ 3        ┆ -0.456   │
        └──────┴──────────┴──────────┘

        # 使用卡方分箱
        woe = WOE(
            cols=['income'],
            target_col='is_default',
            binning_method='chi2',
            chi_merge_threshold=0.1
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌────────┬──────────┬──────────┐
        │ income ┆ income_bin  ┆ income_woe  │
        │ ---    ┆ ---         ┆ ---         │
        │ f64    ┆ i64         ┆ f64         │
        ╞═══════╪═══════════╪═══════════╡
        │ 50000  ┆ 2           ┆ 0.123       │
        │ 60000  ┆ 3           ┆ -0.456      │
        └────────┴──────────┴──────────┘
        ```
    """
    def __init__(
        self, 
        cols, 
        target_col: str,
        binning_method: str = 'quantile',
        n_bins: int = 10,
        bins: Optional[Dict[str, List[float]]] = None,
        min_samples: float = 0.05,
        max_bins: int = 50,
        chi_merge_threshold: float = 0.1,
        save_ori_col: bool = False,
        save_bin_id_col: bool = False,
        save_iv_col: bool = False
    ):
        super().__init__(n_outputs=1)
        self.cols = cols if isinstance(cols, list) else [cols]
        self.target_col = target_col
        self.binning_params = {
            'method': binning_method,
            'n_bins': n_bins,
            'min_samples': min_samples,
            'max_bins': max_bins,
            'chi_merge_threshold': chi_merge_threshold,
        }
        self.custom_bins = bins or {}
        self.save_ori_col = save_ori_col
        self.save_bin_id_col = save_bin_id_col
        self.save_iv_col = save_iv_col

        if not len(self.cols) and self.custom_bins:
            self.cols = list(self.custom_bins.keys())

    def _calculate_woe(self, bin_stats: Dict) -> tuple:
        """计算WOE值和IV值
        
        Returns:
            tuple: (woe_dict, bin_iv_dict, total_iv)
            - woe_dict: 每个分箱的WOE值
            - bin_iv_dict: 每个分箱的IV值
            - total_iv: 总IV值
        """
        total_pos = sum(stats['target_count'] for stats in bin_stats.values())
        total_neg = sum(stats['count'] - stats['target_count'] for stats in bin_stats.values())
        
        woe_dict = {}
        bin_iv_dict = {}
        total_iv = 0
        
        for bin_label, stats in bin_stats.items():
            pos = stats['target_count']
            neg = stats['count'] - pos
            
            # 使用平滑处理
            pos_rate = (pos + 0.5) / (total_pos + 0.5)  # 好样本占比
            neg_rate = (neg + 0.5) / (total_neg + 0.5)  # 坏样本占比
            
            # 计算该分箱的WOE
            woe = np.log(pos_rate / neg_rate)
            woe_dict[bin_label] = float(woe)
            
            # 计算该分箱的IV
            bin_iv = (pos_rate - neg_rate) * woe
            bin_iv_dict[bin_label] = float(bin_iv)
            
            # 累加总IV
            total_iv += bin_iv
            
        return woe_dict, bin_iv_dict, float(total_iv)

    def _plot_woe_summary(self, woe_summary: Dict):
        """使用pyecharts可视化WOE分箱结果和IV值"""
        from pyecharts import options as opts
        from pyecharts.charts import Bar, Line
        
        all_summary = []

        # 创建组合图表
        line_iv = Line()
        features = list(woe_summary.keys())
        iv_values = [round(summary['total_iv'], 3) for summary in woe_summary.values()]
        woe_values = []
        for summary in woe_summary.values():
            bin_woes = [stat['woe'] for stat in summary['bin_stats']]
            avg_woe = round(sum(bin_woes) / len(bin_woes), 3)
            woe_values.append(avg_woe)

        # 添加IV值曲线
        line_iv.add_xaxis(features)
        line_iv.add_yaxis(
            "IV值",
            iv_values,
            symbol_size=8,
            label_opts=opts.LabelOpts(position="top", is_show=True),
            linestyle_opts=opts.LineStyleOpts(width=3),
            # markpoint_opts=opts.MarkPointOpts(
            #     data=[
            #         opts.MarkPointItem(type_="max", name="最大值"),
            #         opts.MarkPointItem(type_="min", name="最小值")
            #     ]
            # )
        )

        # 添加WOE值曲线
        line_iv.add_yaxis(
            "WOE均值",
            woe_values,
            symbol_size=8,
            label_opts=opts.LabelOpts(position="bottom"),
            linestyle_opts=opts.LineStyleOpts(width=3),
            # markpoint_opts=opts.MarkPointOpts(
            #     data=[
            #         opts.MarkPointItem(type_="max", name="最大值"),
            #         opts.MarkPointItem(type_="min", name="最小值")
            #     ]
            # )
        )

        line_iv.set_global_opts(
            title_opts=opts.TitleOpts(title="特征IV值和WOE值分布"),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(rotate=45),
                name="特征"
            ),
            yaxis_opts=opts.AxisOpts(
                name="值",
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross"
            ),
            legend_opts=opts.LegendOpts(pos_top="5%")
        )

        all_summary.append({"IV和WOE分布": line_iv.dump_options_with_quotes()})

        # 每个特征的WOE和分箱IV值可视化
        for col, summary in woe_summary.items():
            # 准备数据
            bin_ranges = [f"{stat['bin_id']}:{stat['bin_range']}" for stat in summary['bin_stats']]
            counts = [stat['count'] for stat in summary['bin_stats']]
            woe_values = [round(stat['woe'], 3) for stat in summary['bin_stats']]
            target_rates = [round(stat['target_rate'] * 100, 3) for stat in summary['bin_stats']]
            bin_iv_values = [round(stat['iv'], 3) for stat in summary['bin_stats']]  # 每个分箱的IV值
            
            # 创建图表
            bar = Bar(
                init_opts=opts.InitOpts(
                    width="900px",
                    height="500px"
                )
            )
            bar.add_xaxis(bin_ranges)
            
            # 添加样本数柱状图
            bar.add_yaxis(
                "样本数",
                counts,
                yaxis_index=0,
                label_opts=opts.LabelOpts(is_show=True, color='black'),
                itemstyle_opts=opts.ItemStyleOpts(opacity=0.4)
            )
            
            # 添加目标率折线图
            line = Line()
            line.add_xaxis(bin_ranges)
            line.add_yaxis(
                "目标率(%)",
                target_rates,
                yaxis_index=2,
                symbol_size=8,
                is_symbol_show=True,
                label_opts=opts.LabelOpts(is_show=True),
                linestyle_opts=opts.LineStyleOpts(width=3),
                z_level=2
            )
            
            # 添加分箱IV值折线图
            line2 = Line()
            line2.add_xaxis(bin_ranges)
            line2.add_yaxis(
                "分箱IV值",
                bin_iv_values,
                yaxis_index=3,
                symbol_size=8,
                is_symbol_show=True,
                label_opts=opts.LabelOpts(is_show=True, position="top"),
                linestyle_opts=opts.LineStyleOpts(width=2),
                z_level=2
            )

            # 添加WOE值折线图
            line3 = Line()
            line3.add_xaxis(bin_ranges)
            line3.add_yaxis(
                "WOE值",
                woe_values,
                yaxis_index=1,
                symbol_size=8,  # 增大点的大小
                is_symbol_show=True,  # 显示数据点
                label_opts=opts.LabelOpts(is_show=True),
                itemstyle_opts=opts.LineStyleOpts(width=3),
                z_level=2
            )
            
            # 组合图表
            bar.overlap(line)
            bar.overlap(line2)
            bar.overlap(line3)
            
            # 设置全局配置
            bar.set_global_opts(
                title_opts=opts.TitleOpts(
                    subtitle=f"分箱数: {len(bin_ranges)}\n总样本数: {sum(counts)}\n特征IV值: {summary['total_iv']:.4f}"
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    axislabel_opts=opts.LabelOpts(rotate=45, interval=0, margin=8)
                ),
                yaxis_opts=opts.AxisOpts(
                    name="样本数",
                    position="left",
                    name_gap=35,
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                datazoom_opts=[opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%", pos_left="center")
            )
            
            # 添加额外的y轴
            bar.extend_axis(
                yaxis=opts.AxisOpts(
                    name="WOE值",
                    position="right",
                    offset=0,
                    name_gap=15,
                    splitline_opts=opts.SplitLineOpts(is_show=False)
                )
            )
            bar.extend_axis(
                yaxis=opts.AxisOpts(
                    name="目标率(%)",
                    position="right",
                    offset=45,
                    name_gap=35,
                    splitline_opts=opts.SplitLineOpts(is_show=False)
                )
            )
            bar.extend_axis(
                yaxis=opts.AxisOpts(
                    name="IV值",
                    position="right",
                    offset=90,
                    name_gap=15,
                    splitline_opts=opts.SplitLineOpts(is_show=False)
                )
            )
            
            all_summary.append({f"{col}": bar.dump_options_with_quotes()})
        return all_summary

    def forward(self, lf: pl.LazyFrame):
        """转换数据
        
        将输入特征转换为WOE编码。对每个特征列，会生成两个新列：
        - {col}_bin: 分箱编号（从0开始的整数）
        - {col}_woe: WOE编码值
        
        Args:
            lf: 输入的LazyFrame或DataFrame
            
        Returns:
            转换后的LazyFrame，包含原始列（如果recover_ori_col=True）和新生成的分箱列、WOE列
            
        Examples:
            >>> woe = WOE(cols=['age'], target_col='is_default')
            >>> df_transformed = woe.forward(df)
            >>> df_transformed.collect()
            shape: (1000, 3)
            ┌──────┬──────────┬──────────┐
            │ age  ┆ age_bin  ┆ age_woe  │
            │ ---  ┆ ---      ┆ ---      │
            │ f64  ┆ i64      ┆ f64      │
            ╞══════╪══════════╪══════════╡
            │ 25.0 ┆ 2        ┆ 0.123    │
            │ 35.0 ┆ 3        ┆ -0.456   │
            └──────┴──────────┴──────────┘
        """
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        
        target = lf.select(pl.col(self.target_col)).collect().to_numpy().flatten()
        woe_summary = {}

        iv_00_01 = []
        iv_01_02 = []
        iv_02_03 = []
        iv_03_04 = []
        iv_04_05 = []
        iv_05_10 = []

        for i, col in enumerate(self.cols):
            self.logger.info(f"正在处理特征{col}, 进度: {i + 1}/{len(self.cols)}")
            values = lf.select(pl.col(col)).collect().to_numpy().flatten()
            
            # 创建并拟合分箱器
            binner = Binning(
                custom_bins=self.custom_bins.get(col),
                **self.binning_params
            )
            binner.fit(values, target)
            
            # 获取分箱编号和对应的标签
            bin_indices = binner.transform(values)
            bin_labels = [binner.binning_result.bin_indices[idx] for idx in bin_indices]
            
            # 计算WOE值映射和IV值
            woe_map, bin_iv_map, total_iv = self._calculate_woe(binner.binning_result.bin_stats)
            woe_values = [woe_map[label] for label in bin_labels]
            iv_values = [bin_iv_map[label] for label in bin_labels]  # 获取每个样本对应的IV值
            
            if total_iv > 0.0 and total_iv <= 0.1:
                iv_00_01.append(col)
            elif total_iv > 0.1 and total_iv <= 0.2:
                iv_01_02.append(col)
            elif total_iv > 0.2 and total_iv <= 0.3:
                iv_02_03.append(col)
            elif total_iv > 0.3 and total_iv <= 0.4:
                iv_04_05.append(col)
            elif total_iv > 0.5 and total_iv <= 1.0:
                iv_05_10.append(col)
            
            # 构建特征的分箱统计信息
            woe_summary[col] = {
                'bin_edges': binner.binning_result.bins,
                'total_iv': total_iv,
                'bin_stats': [
                    {
                        'bin_id': bin_id,
                        'bin_label': bin_label,
                        'bin_range': bin_label,
                        'woe': woe_map[bin_label],
                        'iv': bin_iv_map[bin_label],  # 该分箱的IV值
                        'count': binner.binning_result.bin_stats[bin_label]['count'],
                        'target_count': binner.binning_result.bin_stats[bin_label]['target_count'],
                        'target_rate': (lambda count, target_count: 
                            target_count / count if count != 0 else 0)(
                                binner.binning_result.bin_stats[bin_label]['count'],
                                binner.binning_result.bin_stats[bin_label]['target_count']
                            ),
                        'min': binner.binning_result.bin_stats[bin_label]['min'],
                        'max': binner.binning_result.bin_stats[bin_label]['max'],
                        'mean': binner.binning_result.bin_stats[bin_label]['mean']
                    }
                    for bin_id, bin_label in enumerate(binner.binning_result.bin_labels)
                ]
            }
            
            # 创建WOE和IV表达式
            woe_expr = (
                pl.col(col)
                .replace(dict(zip(values, woe_values)))
                .cast(pl.Float64)
                .alias(f"{col}_woe")
            )
            
            # 添加分箱编号、WOE列和IV列
            lf = lf.with_columns([woe_expr])
            if self.save_bin_id_col:
                bin_expr = (
                    pl.col(col)
                    .replace(dict(zip(values, bin_indices)))
                    .cast(pl.Int64)
                    .alias(f"{col}_bin")
                )
                lf = lf.with_columns([bin_expr])
            if self.save_iv_col:
                iv_expr = (
                    pl.col(col)
                    .replace(dict(zip(values, iv_values)))
                    .cast(pl.Float64)
                    .alias(f"{col}_iv")
                )
                lf = lf.with_columns([iv_expr])
            
            # 如果不保留原始列，则删除
            if not self.save_ori_col:
                lf = lf.drop(col)
        
        self.summary = self._plot_woe_summary(woe_summary)

        self.logger.info(f"IV值在(0, 0.1]的特征【不推荐使用】: {iv_00_01}")
        self.logger.info(f"IV值在(0.1, 0.2]的特征【凑合使用】: {iv_01_02}")
        self.logger.info(f"IV值在(0.2, 0.3]的特征【推荐使用】: {iv_02_03}")
        self.logger.info(f"IV值在(0.3, 0.4]的特征【推荐使用】: {iv_03_04}")
        self.logger.info(f"IV值在(0.4, 0.5]的特征【推荐使用】: {iv_04_05}")
        self.logger.info(f"IV值在(0.5, 1.0]的特征【警惕使用】: {iv_05_10}")
        
        return lf