from pandas import DataFrame
from dags.stage import BaseStage


class FilterStage(BaseStage):
    """
    FilterStage 类用于对 DataFrame 进行过滤操作。

    这个类继承自 BaseStage，用于在数据处理管道中执行数据过滤操作。
    它可以根据指定的条件（如列名、字符串匹配或正则表达式）来选择或排除 DataFrame 中的特定列或行。

    属性:
        items (list): 要选择的标签列表。
        like (str): 用于基于模糊匹配选择标签的字符串。
        regex (str): 用于基于正则表达式选择标签的字符串。
        axis (int): 指定过滤操作的轴，0 表示行，1 表示列。

    方法:
        forward(df: DataFrame) -> DataFrame: 执行过滤操作并返回过滤后的 DataFrame。
    """

    def __init__(self, items: list = None, like: str = None, regex: str = None, axis: int = None):
        super().__init__()
        self.items = items
        self.list = list
        self.like = like
        self.regex = regex
        self.axis = axis

    def forward(self, df: DataFrame) -> DataFrame:
        return df.filter(items=self.items, like=self.like, regex=self.regex, axis=self.axis)
