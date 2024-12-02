import polars as pl
import numpy as np
from dags.stage import CustomStage


class DealNan(CustomStage):

    def __init__(self, col_method: list = []):
        super().__init__(n_outputs=1)
        # 列处理方法 [{'col': 'col_name', 'method': 'mean', 'fill_value': 0}]
        self.col_method = col_method

    def forward(self, lf: pl.LazyFrame):

        if len(self.col_method) == 0:
            self.logger.warn("您未设置列处理方法，将跳过处理")
            return lf
        
        types = lf.collect_schema()

        for method_info in self.col_method:
            self.logger.info(f"开始处理列{method_info['col']}, 处理方法为{method_info['method']}, 自定义填充值为{method_info['fill_value']}")
            col = method_info['col']
            method = method_info['method']
            fill_value = method_info['fill_value']

            if types[col] == pl.String and method in ['mean', 'min', 'max', 'median', 'mode']:
                self.logger.warn(f"您设置的列处理方法为{method}, 但列{col}为字符串类型, 不支持该操作, 将跳过处理")
                continue

            if types[col] == pl.String:
                lf = lf.with_columns(pl.col(col).replace("", None))

            if method == 'mean':
                # 计算非NaN值的平均值
                mean_value = lf.select(pl.col(col).filter(pl.col(col).is_not_nan()).mean()).collect().item()
                lf = lf.with_columns(pl.col(col).fill_nan(mean_value).fill_null(mean_value))
            elif method == 'min':
                min_value = lf.select(pl.col(col).nan_min()).collect().item()
                lf = lf.with_columns(pl.col(col).fill_nan(min_value))
            elif method == 'max':
                max_value = lf.select(pl.col(col).nan_max()).collect().item()
                lf = lf.with_columns(pl.col(col).fill_nan(max_value))
            elif method == 'median':
                median_value = lf.select(pl.col(col).median()).collect().item()
                lf = lf.with_columns(pl.col(col).fill_nan(median_value))
            elif method == 'mode':
                mode_value = lf.select(pl.col(col).mode()).collect().item()
                lf = lf.with_columns(pl.col(col).fill_nan(mode_value))
            elif method == 'custom':
                if types[col] == pl.String:
                    # "" None null 都填充为fill_value
                    lf = lf.with_columns(pl.col(col).fill_null(fill_value))
                else:
                    lf = lf.with_columns(pl.col(col).fill_nan(fill_value).fill_null(fill_value))
            elif method == 'drop':
                # 使用filter来删除包含nan或null的行
                if types[col] == pl.String:
                    filter_expr = ~pl.col(col).is_null()
                else:
                    filter_expr = ~pl.col(col).is_null() & ~pl.col(col).is_nan()
                lf = lf.filter(filter_expr)
            else:
                raise ValueError(f"不支持的列处理方法: {method}")

        return lf

        
