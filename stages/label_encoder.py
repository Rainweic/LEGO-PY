import uuid
import polars as pl
from pyecharts.charts import Tree
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from dags.stage import CustomStage


class CastStage(CustomStage):
    """
    用于对LazyFrame中的列进行值映射的自定义Stage。

    参数:
        col_name (str): 需要进行映射的列名。
        map (dict): 映射字典，键为原始值，值为映射后的值。
        recover_ori_col (bool, 可选): 是否覆盖原始列名。默认为True。
        out_col_name (str, 可选): 如果覆盖原始列名，该参数生效。

    方法:
        forward(df: pl.LazyFrame) -> pl.LazyFrame:
            对输入的LazyFrame进行值映射操作。

    """

    def __init__(self, col_name: str, map: str, recover_ori_col: bool = True, out_col_name: str = None):
        super().__init__(n_outputs=1)
        self.col_name = col_name
        self.map = eval(map)
        self.recover_ori_col = recover_ori_col
        self.out_col_name = out_col_name

    def forward(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        对输入的LazyFrame进行值映射操作。

        参数:
            df (pl.LazyFrame): 输入的LazyFrame。

        返回:
            pl.LazyFrame: 经过值映射操作后的LazyFrame。

        """

        # 将映射字典转换为 SQL CASE WHEN 语句
        case_when_statements = " ".join(
            [f"""WHEN {self.col_name} == {k} THEN {v}""" for k, v in self.map.items()]
        )
        
        if self.recover_ori_col:
            tmp_out_col_name = "casewhen" + str(uuid.uuid4().hex[:8])
            sql_query = f"""
            SELECT 
                *,
                (CASE {case_when_statements} ELSE {self.col_name} END) AS {tmp_out_col_name}
            FROM self
            """.strip()
            self.logger.info(sql_query)
            df = df.lazy().sql(sql_query).drop(pl.col(self.col_name)).rename({tmp_out_col_name: self.col_name})

        else:
            sql_query = f"""
            SELECT 
                *,
                (CASE {case_when_statements} ELSE {self.col_name} END) AS {self.out_col_name}
            FROM self
            """.strip()
            self.logger.info(sql_query)
            df = df.lazy().sql(sql_query)

        return df
    

class LabelEncoder(CustomStage):

    def __init__(self, cols, replace_original=False):
        super().__init__(n_outputs=1)
        self.cols = cols
        self.replace_original = replace_original

    def forward(self, lf: pl.LazyFrame):
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()

        # 初始化特征值映射数据结构
        mapping_data = {
            "name": "特征值映射",
            "children": []
        }
        
        # 对每一列进行编码
        for col in self.cols:

            # 检查列的数据类型
            col_type = lf.select(pl.col(col)).collect().dtypes[0]
            
            # 如果不是数值类型,跳过该列
            if not isinstance(col_type, pl.String):
                self.logger.warning(f"列 {col} 不是String类型,跳过label encoding")
                continue

            # 获取唯一值并排序
            unique_vals = lf.select(pl.col(col)).collect().to_series().unique().sort()
            
            # 创建映射字典 {原值: 编码值}
            mapping = {str(val): idx for idx, val in enumerate(unique_vals) if val is not None}
            
            # 构建映射数据结构
            col_mapping = {
                "name": col,
                "children": [
                    {"name": f"{k}--->{v}"} for k, v in mapping.items()
                ]
            }
            mapping_data["children"].append(col_mapping)
            
            # 构建CASE WHEN语句，添加ELSE子句处理空值和未匹配值
            case_when = " ".join([f"WHEN {col} == '{k}' THEN {v}" for k, v in mapping.items()])
            
            # 应用编码
            if self.replace_original:
                # 如果替换原始列,直接覆盖原列
                sql = f"""
                SELECT 
                    *, 
                    (CASE {case_when} ELSE -1 END) as tmp_{col}_encoded 
                FROM self
                """
                lf = lf.lazy().sql(sql).drop(col).rename({f"tmp_{col}_encoded": col})
            else:
                # 保留原始列,添加新的编码列
                sql = f"""
                SELECT 
                    *,
                    (CASE {case_when} ELSE -1 END) as {col}_encoded 
                FROM self
                """
                lf = lf.lazy().sql(sql)
            
            self.logger.info(f"Encoding column {col} with SQL: {sql}")
            
        # Summary
        tree = (
            Tree(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add("", mapping_data, collapse_interval=2)
            .set_global_opts(title_opts=opts.TitleOpts(title="特征值映射关系"))
            .dump_options_with_quotes()
        )

        self.summary.append({"特征映射": tree})
            
        return lf
