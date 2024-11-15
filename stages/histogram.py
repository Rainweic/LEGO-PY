import json
import polars as pl
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
from stages.utils.binning import Binning, binning_categorical
from dags.stage import CustomStage


class Histogram(CustomStage):

    def __init__(self, cols: list = [], method: str = 'equal_freq', n_bins: int = 10, custom_bins: dict = None):
        super().__init__(n_outputs=0)
        self.cols = cols
        self.method = method
        self.n_bins = n_bins
        self.custom_bins = custom_bins

        if isinstance(custom_bins, str):
            self.custom_bins = json.loads(custom_bins)

    def forward(self, data: pl.LazyFrame):
        # 使用collect_schema()来避免性能警告
        schema = data.collect_schema()
        df = data.collect()
        histograms = []
        
        if not self.cols:
            self.logger.info("No columns specified, using all columns")
            self.cols = data.columns

        for col in self.cols:
            # 获取列的数据类型
            dtype = schema[col]
            
            # 创建单列DataFrame以提高性能
            f_data = df.select(pl.col(col))
            
            # 判断是否为数值类型
            if dtype not in [pl.Categorical, pl.Utf8, pl.String]:
                histograms.append({col: self._process_numeric_column(f_data, col).dump_options_with_quotes()})
            else:
                histograms.append({col: self._process_categorical_column(f_data, col).dump_options_with_quotes()})
        
        self.summary = histograms

    def _process_numeric_column(self, df: pl.DataFrame, column: str):
        # 使用Binning类进行分箱
        binning = Binning(
            method=self.method,
            n_bins=self.n_bins,
            custom_bins=self.custom_bins.get(column) if self.custom_bins else None,
            min_samples=0.05  # 设置最小样本比例为5%
        )
        
        values = df[column].to_numpy()
        
        if self.custom_bins and column in self.custom_bins:
            # 使用自定义分箱
            bins = self.custom_bins[column]
            # 确保包含最小值和最大值
            if bins[0] > values.min():
                bins.insert(0, values.min())
            if bins[-1] < values.max():
                bins.append(values.max())
            
            # 计算每个区间的频数
            counts, edges = np.histogram(values, bins=bins)
            bin_labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges)-1)]
            
        else:
            # 使用Binning类的分箱方法
            bin_ids, bin_labels = binning.fit_transform(values, return_labels=True)  # 修改这里，要求返回标签
            statistic = pl.DataFrame({"label": bin_labels, "bin_ids": bin_ids}).group_by("label").count().sort("label")
            bin_labels = statistic["label"].to_list()
            counts = statistic["count"].to_list()

        # 创建直方图
        chart = self._create_histogram(
            values=counts,
            labels=bin_labels,
            title=f"Histogram of {column}",
            x_label=column,
            y_label="Frequency",
            is_numeric=True
        )
        
        return chart

    def _process_categorical_column(self, df: pl.DataFrame, column: str):
        # 计算每个类别的频率
        bins, counts = binning_categorical(df[column], self.n_bins)
        
        # 创建直方图
        chart = self._create_histogram(
            values=counts,
            labels=bins,
            title=f"Histogram of {column}",
            x_label=column,
            y_label="Frequency",
            is_numeric=False
        )
        
        return chart

    def _create_histogram(self, values, labels, title, x_label, y_label, is_numeric=True):
        bar = Bar()
        bar.add_xaxis(labels)
        bar.add_yaxis(
            "频率", 
            values if isinstance(values, list) else values.tolist(), 
            label_opts=opts.LabelOpts(is_show=False)
        )
        
        # 设置图表属性
        bar.set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                subtitle=f"Total categories: {len(labels)}" if not is_numeric else None
            ),
            xaxis_opts=opts.AxisOpts(
                name=x_label,
                axislabel_opts=opts.LabelOpts(rotate=45 if not is_numeric else 0)
            ),
            yaxis_opts=opts.AxisOpts(name=y_label),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)
            ]
        )
        
        return bar
