import logging
import polars as pl
import pyarrow.orc as orc
from dags.stage import CustomStage


class CSVReadStage(CustomStage):

    def __init__(self, path, select_cols: list = []):
        super().__init__(n_outputs=1)
        self.path = path
        self.select_cols = select_cols

    def forward(self):
        df = pl.scan_csv(self.path)
        if self.select_cols:
            df = df.select(self.select_cols)
        return df
    

class ORCReadStage(CSVReadStage):

    def forward(self):
        try:
            # 使用 pyarrow 读取 ORC 文件
            orc_file = orc.ORCFile(self.path)
        except BaseException as e:
            self.logger.error(f"Reading file {self.path} error: {e}")
            raise e
        
        df_list = []

        # 分块读取 ORC 文件
        for i in range(0, orc_file.nstripes):
            batch = orc_file.read_stripe(i, columns=self.select_cols)
            # 将 pyarrow Table 转换为 polars DataFrame 并追加到 df
            df_list.append(pl.from_arrow(batch).lazy())

        self.logger.info("Starting concat lazy dataframe")
        df = pl.concat(df_list, how="vertical_relaxed")

        return df