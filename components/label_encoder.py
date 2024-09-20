import pandas as pd

from dags.stage import BaseStage


class CastStage(BaseStage):
    """
    CastStage 类用于将DataFrame中的特定列进行映射转换。

    这个类继承自BaseStage,用于在数据处理管道中执行特征值的映射转换操作。
    它可以将一个列中的值根据提供的映射字典进行转换,通常用于类别编码或值的重映射。

    属性:
        feature_name (str): 需要进行转换的特征(列)名称。
        map (dict): 用于转换的映射字典,键为原始值,值为映射后的新值。

    方法:
        forward(df: pd.DataFrame) -> pd.DataFrame: 执行映射转换操作。
    """

    def __init__(self, feature_name: str, map: dict):
        """
        初始化CastStage实例。

        参数:
            feature_name (str): 需要进行转换的特征(列)名称。
            map (dict): 用于转换的映射字典,键为原始值,值为映射后的新值。
        """
        super().__init__()
        self.feature_name = feature_name
        self.map = map

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对输入的DataFrame执行映射转换操作。

        将指定列(feature_name)中的值根据提供的映射字典(map)进行转换。

        参数:
            df (pd.DataFrame): 输入的DataFrame。

        返回:
            pd.DataFrame: 转换后的DataFrame。
        """
        df[self.feature_name] = df[self.feature_name].map(self.map)
        return df
