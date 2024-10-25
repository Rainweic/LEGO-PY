import polars as pl
import numpy as np
from pyecharts.charts import Pie
from pyecharts import options as opts
from dags.stage import CustomStage


class Statistics(CustomStage):

    def __init__(self, cols=[], n_bins=10):
        super().__init__(n_outputs=1)
        self.cols = cols
        self.n_bins = n_bins

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
                # 对数值型数据进行等距分箱
                min_val, max_val = np.min(col_data), np.max(col_data)
                bins = np.linspace(min_val, max_val, self.n_bins + 1)
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
            # print(col_dtype)
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
        self.logger.info(col_stats.collect())
        
        import json
        json.dumps(self.summary)

        return stats_out
