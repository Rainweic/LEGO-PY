import os
import pandas as pd
import polars as pl
import hashlib
import logging
import pyarrow.orc as orc
import pyarrow.dataset as ds
from utils.hdfs_v3 import download
from dags.stage import CustomStage, stage
from tqdm import tqdm


CACHE_PATH = "./cache/download"


def hdfs_download(path: str, overwrite: bool, logger):
    # 确保缓存目录存在
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    # 使用路径的哈希值创建唯一的文件名
    path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
    base_name = os.path.basename(path)
    unique_name = f"{path_hash}_{base_name}"

    # 构建本地缓存文件路径
    local_path = os.path.join(CACHE_PATH, unique_name)
    logger.info(f"Download: {path} -> {local_path}")

    if os.path.exists(local_path):
        if overwrite:
            download(
                path, local_path, "obs://lts-bigdata-hive-obs-prod/", 10, True, "hadoop"
            )
            logger.info("下载完成")
        else:
            logger.warn(f"{local_path}已仍存在，不再下载")
    else:
        download(
            path, local_path, "obs://lts-bigdata-hive-obs-prod/", 10, True, "hadoop"
        )
        logger.info("下载完成")
    return local_path


class HDFSCSVReadStage(CustomStage):

    def __init__(
        self, path: str, select_cols: list = [], overwrite: bool = False, *args, **kwargs
    ):
        super().__init__(n_outputs=1)
        self.path = path
        self.overwrite = overwrite
        self.select_cols = select_cols

    @property
    def reader(self):
        return pl.scan_csv

    @property
    def file_type(self):
        return ".csv"

    def forward(self, *args, **kwargs) -> pl.LazyFrame:

        local_path = hdfs_download(self.path, self.overwrite, self.logger)

        df_list = []
        total_files = sum([len(files) for _, _, files in os.walk(local_path)])

        with tqdm(total=total_files, desc="Reading df", unit="file") as pbar:
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file.endswith(self.file_type):
                        file_path = os.path.join(root, file)
                        try:
                            df = self.reader(file_path)
                            if self.select_cols:
                                df = df.select(self.select_cols)
                            df_list.append(df)
                        except pd.errors.ParserError as e:
                            self.logger.error(
                                f"stage {self.name} read {file_path} error: {e}"
                            )
                            raise e
                        finally:
                            pbar.update(1)

        if df_list:
            self.logger.info("Starting concat lazy dataframe")
            df = pl.concat(df_list, how="vertical_relaxed")
        else:
            raise RuntimeError(f"Can not find any {self.file_type} files")

        return df


class HDFSORCReadStage(HDFSCSVReadStage):

    @property
    def file_type(self):
        return ".orc"
    
    def forward(self, *args, **kwargs) -> pl.LazyFrame:
        
        local_path = hdfs_download(self.path, self.overwrite, self.logger)

        df_list = []

        # 获取所有 ORC 文件的路径
        orc_files = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(self.file_type):
                    orc_files.append(os.path.join(root, file))

        if not orc_files:
            raise RuntimeError(f"Can not find any {self.file_type} files")

        # 使用 tqdm 显示进度条
        for file_path in tqdm(orc_files, desc="Reading ORC files"):
            try:
                # 使用 pyarrow 读取 ORC 文件
                orc_file = orc.ORCFile(file_path)
            except BaseException as e:
                self.logger.error(f"Reading file {file_path} error: {e}")
                raise e
            # 获取 ORC 文件的 schema
            schema = orc_file.schema
            # 初始化一个空的 polars DataFrame
            df = pl.DataFrame(schema.empty_table().to_pandas())
            
            # 分块读取 ORC 文件
            for i in range(0, orc_file.nstripes):
                batch = orc_file.read_stripe(i, columns=self.select_cols)
                # 将 pyarrow Table 转换为 polars DataFrame 并追加到 df
                df_list.append(pl.from_arrow(batch).lazy())

        self.logger.info("Starting concat lazy dataframe")
        df = pl.concat(df_list, how="vertical_relaxed")

        return df


# @stage(n_outputs=1)
# def hdfs_csv_reader(path: str, *args) -> pd.DataFrame:

#     local_path = hdfs_download(path)

#     # 遍历目录，全部读取csv文件并concat
#     df_list = []
#     for root, dirs, files in os.walk(local_path):
#         for file in files:
#             if file.endswith('.csv'):
#                 file_path = os.path.join(root, file)
#                 df = pd.read_csv(file_path)
#                 df_list.append(df)

#     if df_list:
#         df = pd.concat(df_list, ignore_index=True)
#     else:
#         df = pd.DataFrame()  # 如果没有找到CSV文件，返回空的DataFrame

#     return df


# @stage
# def hdfs_orc_reader(path: str, *args) -> pd.DataFrame:

#     local_path = hdfs_download(path)

#     # 遍历目录，全部读取csv文件并concat
#     df_list = []
#     for root, dirs, files in os.walk(local_path):
#         for file in files:
#             if file.endswith('.csv'):
#                 file_path = os.path.join(root, file)
#                 df = pd.read_orc(file_path)
#                 df_list.append(df)

#     if df_list:
#         df = pd.concat(df_list, ignore_index=True)
#     else:
#         df = pd.DataFrame()  # 如果没有找到CSV文件，返回空的DataFrame

#     return df


# 机器不连spark集群，估计用不到
# @stage
# def hdfs_sql(sql: str) -> pd.DataFrame:
#     spark = SparkSession.builder.getOrCreate()
#     return spark.sql(sql).toPandas()
