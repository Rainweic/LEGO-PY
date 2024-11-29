import numpy as np
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
from pyecharts import options as opts
from pyecharts.charts import Bar


def plot_proba_distribution(array_A: np.ndarray, array_B: np.ndarray, n_bins: int = 50, title: str = "数据分布对比"):
    """使用pyecharts绘制概率分布对比图
    
    Args:
        array_A: 实验组数据数组
        array_B: 对照组数据数组
        n_bins: 分箱数量（仅用于数值型数据），默认50
    """
    # 检查数据类型
    is_numeric = np.issubdtype(array_A.dtype, np.number)
    
    if is_numeric:
        # 数值类型处理逻辑
        min_val = min(array_A.min(), array_B.min())
        max_val = max(array_A.max(), array_B.max())
        edges = np.linspace(min_val, max_val, n_bins + 1)
        
        hist_A, _ = np.histogram(array_A, bins=edges, density=True)
        hist_B, _ = np.histogram(array_B, bins=edges, density=True)
        
        x_points = (edges[:-1] + edges[1:]) / 2
        x_labels = [f"{x:.3f}" for x in x_points]
    else:
        # 字符串类型处理逻辑
        unique_values = np.unique(np.concatenate([array_A, array_B]))
        x_labels = unique_values.tolist()
        
        # 计算每个值的频次并归一化
        hist_A = np.array([np.sum(array_A == val) for val in unique_values]) / len(array_A)
        hist_B = np.array([np.sum(array_B == val) for val in unique_values]) / len(array_B)

    line = Line(
        init_opts=opts.InitOpts(
            theme=ThemeType.LIGHT,
            width="900px",
            height="500px"
        )
    )
        
    line.add_xaxis(xaxis_data=x_labels)
    
    # 添加实验组分布曲线
    line.add_yaxis(
        series_name="实验组",
        y_axis=hist_A.tolist(),
        symbol_size=8,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2)
    )
    
    # 添加对照组分布曲线
    line.add_yaxis(
        series_name="对照组",
        y_axis=hist_B.tolist(),
        symbol_size=8,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2)
    )
    
    # 设置全局选项
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(
            name="Propensity Score",
            name_location="center",
            name_gap=35,
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            name="密度",
            name_location="center",
            name_gap=40,
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    
    return line


def plot_curve(x: np.ndarray, y: np.ndarray, title: str = "曲线图", n_samples: int = 300):
    """使用pyecharts绘制曲线图
    
    Args:
        x: x轴数据数组
        y: y轴数据数组
        title: 图表标题
        n_samples: 采样点数量，默认300
    """
    # 对数据进行采样
    if len(x) > n_samples:
        indices = np.linspace(0, len(x) - 1, n_samples, dtype=int)
        x_sampled = x[indices]
        y_sampled = y[indices]
    else:
        x_sampled = x
        y_sampled = y
    
    # 创建曲线图实例
    line = Line(
        init_opts=opts.InitOpts(
            theme=ThemeType.LIGHT,
            width="900px",
            height="500px"
        )
    )
    
    # 添加数据
    line.add_xaxis(xaxis_data=[f"{x:.3f}" for x in x_sampled])
    line.add_yaxis(
        series_name="",
        y_axis=y_sampled.tolist(),
        symbol_size=8,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2)
    )
    
    # 设置全局选项
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    
    return line


def plot_hist_compare(array_A: np.ndarray, array_B: np.ndarray, n_bins: int = 20, title: str = "数据分布对比"):
    """使用pyecharts绘制分布对比曲线图

    Args:
        array_A: 实验组数据数组
        array_B: 对照组数据数组
        n_bins: 分箱数量，默认20
        title: 图表标题
    """
    # 计算分箱边界
    min_val = min(array_A.min(), array_B.min())
    max_val = max(array_A.max(), array_B.max())
    edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # 计算直方图数据
    hist_A, _ = np.histogram(array_A, bins=edges)
    hist_B, _ = np.histogram(array_B, bins=edges)
    
    # 生成x轴标签
    x_points = (edges[:-1] + edges[1:]) / 2
    x_labels = [f"{x:.2f}" for x in x_points]
    
    # 创建曲线图实例
    line = Line(
        init_opts=opts.InitOpts(
            theme=ThemeType.LIGHT,
            width="900px",
            height="500px"
        )
    )
    
    # 添加数据
    line.add_xaxis(xaxis_data=x_labels)
    line.add_yaxis(
        series_name="实验组",
        y_axis=hist_A.tolist(),
        symbol_size=8,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2)
    )
    line.add_yaxis(
        series_name="对照组",
        y_axis=hist_B.tolist(),
        symbol_size=8,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2)
    )
    
    # 设置全局选项
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(
            name="取值",
            name_location="center",
            name_gap=35,
            axislabel_opts=opts.LabelOpts(rotate=45),
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            name="频次",
            name_location="center",
            name_gap=40,
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside")
        ],
    )
    
    return line


    
