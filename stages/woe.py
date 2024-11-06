import polars as pl
import numpy as np
from typing import Dict, List, Optional
from dags.stage import CustomStage
from stages.utils.binning import Binning


class WOE(CustomStage):
    """WOE(Weight of Evidence)编码转换器
    
    将连续型或分类型特征转换为WOE编码。支持多种分箱方法，包括等频、等宽、卡方等。
    WOE = ln(好样本占比/坏样本占比)，可以将特征转换为与目标变量的相关性度量。

    参数:
        cols (Union[str, List[str]]): 
            需要进行WOE编码的列名，可以是单个字符串或字符串列表
        
        target_col (str): 
            目标变量的列名，必须是二值型变量(0/1)
        
        binning_method (str, 默认='quantile'): 
            分箱方法，支持以下几种:
            - 'equal_width': 等宽分箱
            - 'equal_freq': 等频分箱(quantile)
            - 'chi2': 卡方分箱(基于目标变量)
            - 'kmeans': 基于聚类的分箱
            - 'mdlp': 最小描述长度分箱
        
        n_bins (int, 默认=10): 
            分箱数量，仅在使用等宽、等频、kmeans分箱时有效
        
        bins (Optional[Dict[str, List[float]]], 默认=None): 
            自定义分箱点，格式为 {列名: [分箱点列表]}
            如果某列指定了自定义分箱点，将忽略binning_method和n_bins参数
        
        min_samples (float, 默认=0.05): 
            每个分箱最小样本比例，用于防止过度分箱
        
        max_bins (int, 默认=50): 
            最大分箱数量，用于防止过度分箱
        
        chi_merge_threshold (float, 默认=0.1): 
            卡方分箱时的合并阈值，较大的值会产生更少的分箱
        
        recover_ori_col (bool, 默认=True):
            是否在结果中保留原始特征列
    
    返回:
        对于每个输入特征col，会生成两个新列：
        - {col}_bin: 分箱编号（从0开始的整数）
        - {col}_woe: WOE编码值

    示例:
        ```python
        # 基本使用
        woe = WOE(
            cols=['age', 'income'],
            target_col='is_default',
            binning_method='equal_freq',
            n_bins=10
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌──────┬──────────┬──────────┐
        │ age  ┆ age_bin  ┆ age_woe  │
        │ ---  ┆ ---      ┆ ---      │
        │ f64  ┆ i64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ 25.0 ┆ 2        ┆ 0.123    │
        │ 35.0 ┆ 3        ┆ -0.456   │
        └──────┴──────────┴──────────┘

        # 使用自定义分箱点
        woe = WOE(
            cols=['age'],
            target_col='is_default',
            bins={'age': [0, 18, 25, 35, 50, 65, float('inf')]}
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌──────┬──────────┬──────────┐
        │ age  ┆ age_bin  ┆ age_woe  │
        │ ---  ┆ ---      ┆ ---      │
        │ f64  ┆ i64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ 25.0 ┆ 2        ┆ 0.123    │
        │ 35.0 ┆ 3        ┆ -0.456   │
        └──────┴──────────┴──────────┘

        # 使用卡方分箱
        woe = WOE(
            cols=['income'],
            target_col='is_default',
            binning_method='chi2',
            chi_merge_threshold=0.1
        )
        df_transformed = woe.forward(df)
        >>> df_transformed.collect()
        shape: (1000, 3)
        ┌────────┬──────────┬──────────┐
        │ income ┆ income_bin  ┆ income_woe  │
        │ ---    ┆ ---         ┆ ---         │
        │ f64    ┆ i64         ┆ f64         │
        ╞═══════╪═══════════╪═══════════╡
        │ 50000  ┆ 2           ┆ 0.123       │
        │ 60000  ┆ 3           ┆ -0.456      │
        └────────┴──────────┴──────────┘
        ```
    """
    def __init__(
        self, 
        cols, 
        target_col: str,
        binning_method: str = 'quantile',
        n_bins: int = 10,
        bins: Optional[Dict[str, List[float]]] = None,
        min_samples: float = 0.05,
        max_bins: int = 50,
        chi_merge_threshold: float = 0.1,
        recover_ori_col: bool = True
    ):
        super().__init__(n_outputs=1)
        self.cols = cols if isinstance(cols, list) else [cols]
        self.target_col = target_col
        self.binning_params = {
            'method': binning_method,
            'n_bins': n_bins,
            'min_samples': min_samples,
            'max_bins': max_bins,
            'chi_merge_threshold': chi_merge_threshold,
        }
        self.custom_bins = bins or {}
        self.recover_ori_col = recover_ori_col

        if not len(self.cols) and self.custom_bins:
            self.cols = list(self.custom_bins.keys())

    def _calculate_woe(self, bin_stats: Dict) -> Dict:
        """计算WOE值"""
        total_pos = sum(stats['target_count'] for stats in bin_stats.values())
        total_neg = sum(stats['count'] - stats['target_count'] for stats in bin_stats.values())
        
        woe_dict = {}
        for bin_label, stats in bin_stats.items():
            pos = stats['target_count']
            neg = stats['count'] - pos
            
            # 使用平滑处理
            pos_rate = (pos + 0.5) / (total_pos + 0.5)
            neg_rate = (neg + 0.5) / (total_neg + 0.5)
            
            woe = np.log(pos_rate / neg_rate)
            woe_dict[bin_label] = float(woe)
            
        return woe_dict

    def forward(self, lf: pl.LazyFrame):
        """转换数据
        
        将输入特征转换为WOE编码。对每个特征列，会生成两个新列：
        - {col}_bin: 分箱编号（从0开始的整数）
        - {col}_woe: WOE编码值
        
        Args:
            lf: 输入的LazyFrame或DataFrame
            
        Returns:
            转换后的LazyFrame，包含原始列（如果recover_ori_col=True）和新生成的分箱列、WOE列
            
        Examples:
            >>> woe = WOE(cols=['age'], target_col='is_default')
            >>> df_transformed = woe.forward(df)
            >>> df_transformed.collect()
            shape: (1000, 3)
            ┌──────┬──────────┬──────────┐
            │ age  ┆ age_bin  ┆ age_woe  │
            │ ---  ┆ ---      ┆ ---      │
            │ f64  ┆ i64      ┆ f64      │
            ╞══════╪══════════╪══════════╡
            │ 25.0 ┆ 2        ┆ 0.123    │
            │ 35.0 ┆ 3        ┆ -0.456   │
            └──────┴──────────┴──────────┘
        """
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        
        target = lf.select(pl.col(self.target_col)).collect().to_numpy().flatten()
        
        for col in self.cols:
            values = lf.select(pl.col(col)).collect().to_numpy().flatten()
            
            # 创建并拟合分箱器
            binner = Binning(
                custom_bins=self.custom_bins.get(col),
                **self.binning_params
            )
            binner.fit(values, target)
            
            # 获取分箱编号和对应的标签
            bin_indices = binner.transform(values)  # 现在返回分箱编号
            bin_labels = [binner.binning_result.bin_indices[idx] for idx in bin_indices]
            
            # 计算WOE值映射
            woe_map = self._calculate_woe(binner.binning_result.bin_stats)
            woe_values = [woe_map[label] for label in bin_labels]
            
            # 创建分箱编号和WOE映射表达式
            bin_expr = (
                pl.col(col)
                .replace(
                    dict(zip(values, bin_indices))
                )
                .cast(pl.Int64)  # 确保分箱编号为整数类型
                .alias(f"{col}_bin")
            )
            
            woe_expr = (
                pl.col(col)
                .replace(
                    dict(zip(values, woe_values))
                )
                .cast(pl.Float64)  # 确保WOE值为浮点数类型
                .alias(f"{col}_woe")
            )
            
            # 添加新列
            lf = lf.with_columns([bin_expr, woe_expr])
            
            # 如果不保留原始列，则删除
            if not self.recover_ori_col:
                lf = lf.drop(col)
        
        return lf