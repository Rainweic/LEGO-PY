from pandas import DataFrame
from dags.stage import BaseStage


class CustomFuncStage(BaseStage):
    """
    CustomFuncStage 类用于执行自定义函数对DataFrame进行处理。

    这个类继承自BaseStage，允许用户在数据处理管道中插入自定义的数据处理逻辑。
    它可以接受任何可调用对象（函数或方法）作为处理逻辑，并在forward方法中执行。

    属性:
        custom_func (callable): 用户定义的自定义函数，用于处理DataFrame。该函数只传入一个df参数
        args (tuple): 传递给custom_func的位置参数。
        kwargs (dict): 传递给custom_func的关键字参数。

    方法:
        forward(df: DataFrame) -> DataFrame: 执行自定义函数处理DataFrame。
    """

    def __init__(self, custom_func: callable, *args, **kwargs):
        """
        初始化CustomFuncStage实例。

        参数:
            custom_func (callable): 用户定义的自定义函数，用于处理DataFrame。
            *args: 传递给custom_func的位置参数。
            **kwargs: 传递给custom_func的关键字参数。
        """
        super().__init__()
        self.custom_func = custom_func
        self.args = args
        self.kwargs = kwargs

    def forward(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        """
        将输入的DataFrame传递给自定义函数，并返回处理后的结果。

        参数:
            df (DataFrame): 输入的DataFrame。

        返回:
            DataFrame: 经过自定义函数处理后的DataFrame。
        """
        return self.custom_func(df, *self.args, **self.kwargs)
