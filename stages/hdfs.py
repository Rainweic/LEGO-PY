import os
import pandas as pd
import hashlib
import logging
from utils.hdfs_v3 import download
from dags.stage import CustomStage


CACHE_PATH = "./cache"


def hdfs_download(path: str, overwrite: bool):
    # 确保缓存目录存在
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    
    # 使用路径的哈希值创建唯一的文件名
    path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
    base_name = os.path.basename(path)
    unique_name = f"{path_hash}_{base_name}"
    
    # 构建本地缓存文件路径
    local_path = os.path.join(CACHE_PATH, unique_name)
    logging.info(f"Download: {path} -> {local_path}")
    
    if os.path.exists(local_path):
        if overwrite:
            download(path, local_path, "obs://lts-bigdata-hive-obs-prod/", 10, True, 'hadoop')
        else:
            logging.warn(f"{local_path}已仍存在，不再下载")
    else:
        download(path, local_path, "obs://lts-bigdata-hive-obs-prod/", 10, True, 'hadoop')
    return local_path


class HDFSCSVReadStage(CustomStage):

    def __init__(self, path: str, select_cols: list = [], overwrite: bool = True, *args, **kwargs):
        super().__init__(n_outputs=1)
        self.path = path
        self.overwrite = overwrite
        self.select_cols = select_cols

    @property
    def reader(self):
        return pd.read_csv
    
    @property
    def file_type(self):
        return ".csv"
    
    def forward(self, *args, **kwargs) -> pd.DataFrame:

        local_path = hdfs_download(self.path, self.overwrite)
    
        # 遍历目录，全部读取csv文件并concat
        df_list = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(self.file_type):
                    file_path = os.path.join(root, file)
                    df = self.reader(file_path)
                    if self.select_cols:
                        df = df[self.select_cols]
                    df_list.append(df)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            raise RuntimeError(f"Can not find any {self.file_type} files")
        
        return df
    

class HDFSORCReadStage(HDFSCSVReadStage):

    @property
    def reader(self):
        return pd.read_orc
    
    @property
    def file_type(self):
        return ".orc"
    

# @stage
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