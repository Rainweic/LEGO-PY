import polars as pl
from dags.stage import CustomStage


class Union(CustomStage):

    def __init__(self, drop_duplicate=False, how='vertical_relaxed', maintain_order=False):
        super().__init__(n_outputs=1)
        self.drop_duplicate = drop_duplicate
        self.how = how
        self.maintain_order = maintain_order

    def forward(self, lf1: pl.LazyFrame, lf2: pl.LazyFrame):
        try:
            schema1 = lf1.schema
            schema2 = lf2.schema
            
            # self.logger.info(f"lf1 schema: {schema1}")
            # self.logger.info(f"lf2 schema: {schema2}")

            # 检查列类型是否一致
            common_cols = set(schema1.keys()) & set(schema2.keys())
            for col in common_cols:
                if schema1[col] != schema2[col]:
                    self.logger.warning(f"列 {col} 类型不一致: lf1={schema1[col]}, lf2={schema2[col]}")
                    # 可以尝试进行类型转换
                    if pl.datatypes.is_numeric(schema1[col]) and pl.datatypes.is_numeric(schema2[col]):
                        # 转换为更高精度的类型
                        target_type = max(schema1[col], schema2[col], key=lambda x: x._numpy_dtype.itemsize)
                        lf1 = lf1.with_columns(pl.col(col).cast(target_type))
                        lf2 = lf2.with_columns(pl.col(col).cast(target_type))
            
            result = pl.concat([lf1, lf2], how=self.how)
            
            if self.drop_duplicate:
                self.logger.info("开始去重")
                result = result.select(pl.col("*").unique(maintain_order=self.maintain_order))
            
            return result

        except Exception as e:
            self.logger.error(f"合并失败: {str(e)}")
            self.logger.error(f"lf1 信息: schema={lf1.schema}, shape={lf1.collect().shape}")
            self.logger.error(f"lf2 信息: schema={lf2.schema}, shape={lf2.collect().shape}")
            raise

