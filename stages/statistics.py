import polars as pl
import numpy as np
from pyecharts.charts import Pie
from pyecharts import options as opts
from dags.stage import CustomStage


class Statistics(CustomStage):

    def __init__(self, cols=[], n_bins=10, bin_type='equal_width'):
        super().__init__(n_outputs=1)
        self.cols = cols
        self.n_bins = n_bins
        self.bin_type = bin_type

    def forward(self, df: pl.LazyFrame):

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        for col in self.cols:
            # 获取列数据
            col_data = df.select(pl.col(col)).collect().to_numpy()
            
            # 对于非数值型数据,我们只能进行简单的值计数
            if not np.issubdtype(col_data.dtype, np.number):
                value_counts = df.lazy().select([pl.col(col).value_counts()]).collect()
                data = [(str(item[0][0]), int(item[0][1])) for item in value_counts.to_numpy()]
            else:
                # 数值型数据根据 bin_type 选择分箱方式
                if self.bin_type == 'equal_width':
                    # 等距分箱（原有逻辑）
                    min_val, max_val = np.min(col_data), np.max(col_data)
                    bins = np.linspace(min_val, max_val, self.n_bins + 1)
                elif self.bin_type == 'equal_freq':  # equal_freq
                    # 等频分箱
                    bins = np.percentile(col_data, 
                                      np.linspace(0, 100, self.n_bins + 1))
                else:
                    raise TypeError('分箱类型仅支持[equal_width, equal_freq]')
                
                # 生成区间标签
                labels = [f'{bins[i]:.2f}~{bins[i+1]:.2f}' for i in range(self.n_bins)]
                
                # 统计每个箱的样本数量
                hist, _ = np.histogram(col_data, bins=bins)
                data = list(zip(labels, hist.tolist()))
            
            # print(data)

            # 创建饼图
            pie = (
                Pie()
                .add(f"", data)
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}个"))
            )
            
            self.summary.append({f"{col}": pie.dump_options_with_quotes()})

        stats_out_list = []
        for col in self.cols:
            col_dtype = str(df.collect_schema()[col])
            # 检查列是否全为空值
            is_all_null = df.select(pl.col(col).is_null().all()).collect().item()
            
            if is_all_null:
                # 如果列全为空值，设置特殊的统计值
                col_stats = pl.DataFrame({
                    "特征": [col],
                    "max": [None],
                    "min": [None],
                    "mean": [None],
                    "null_count": [df.select(pl.col(col).null_count()).collect().item()],
                    "n_unique": [0],
                    "std": [None]
                }).lazy()
            else:
                # 正常计算统计值
                col_stats = df.lazy().select([
                    pl.lit(col).alias("特征"),
                    pl.col(col).max().cast(pl.Utf8).alias("max"),
                    pl.col(col).min().cast(pl.Utf8).alias("min"),
                    pl.col(col).mean().cast(pl.Float64).alias("mean"),
                    pl.col(col).null_count().cast(pl.Int64).alias("null_count"),
                    pl.col(col).n_unique().cast(pl.Int64).alias("n_unique"),
                    pl.col(col).std().cast(pl.Float64).alias("std") if col_dtype != "String" else pl.lit(None).cast(pl.Float64).alias("std"),
                ])
            stats_out_list.append(col_stats)

        stats_out: pl.DataFrame = pl.concat(stats_out_list).collect()
        self.logger.info(stats_out)
        
        import json
        json.dumps(self.summary)

        return stats_out
