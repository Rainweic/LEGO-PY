import os
import pandas as pd
from pyspark.sql import SparkSession
from dags.stage import stage, SQLiteStage
from utils.hdfs_v3 import download


CACHE_PATH = "./cache"


def hdfs_download(path: str):
    # 确保缓存目录存在
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    
    # 构建本地缓存文件路径
    local_path = os.path.join(CACHE_PATH, os.path.basename(path))
    
    download(path, local_path, "obs://lts-bigdata-hive-obs-prod/", 10, True, 'hadoop')
    return local_path


class HDFSCSVReader(SQLiteStage):

    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path

    @property
    def reader(self):
        return pd.read_csv
    
    def run(self, *args, **kwargs):

        local_path = hdfs_download(self.path)
    
        # 遍历目录，全部读取csv文件并concat
        df_list = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = self.reader(file_path)
                    df_list.append(df)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            df = pd.DataFrame()  # 如果没有找到CSV文件，返回空的DataFrame
        
        return df
    

class HDFSORCReader(HDFSCSVReader):

    @property
    def reader():
        return pd.read_orc
    

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
@stage
def hdfs_sql(sql: str) -> pd.DataFrame:
    spark = SparkSession.builder.getOrCreate()
    return spark.sql(sql).toPandas()