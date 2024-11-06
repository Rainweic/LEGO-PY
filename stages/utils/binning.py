import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import pandas as pd

@dataclass
class BinningResult:
    """存储分箱结果
    
    Attributes:
        bins: 分箱边界值列表，长度为n_bins+1
        bin_labels: 分箱标签列表，长度为n_bins
        bin_stats: 每个分箱的统计信息字典，包含count、min、max、mean等
        bin_indices: 分箱编号到标签的映射字典
    """
    bins: List[float]  # 分箱边界
    bin_labels: List[str]  # 分箱标签
    bin_stats: Dict  # 每个分箱的统计信息
    bin_indices: Dict[int, str]  # 分箱编号到标签的映射

class Binning:
    """特征分箱类，支持多种分箱方法
    
    支持的分箱方法:
        - equal_width: 等宽分箱，将数据按照相等的区间宽度进行分箱
        - equal_freq: 等频分箱（默认），将数据按照每个箱中样本数量大致相等的原则进行分箱
        - chi2: 卡方分箱，基于卡方检验的自适应分箱方法，适合处理分类目标变量
        - kmeans: KMeans聚类分箱，使用K均值聚类的方法进行分箱
        - mdlp: 最小描述长度分箱，基于信息熵的自适应分箱方法，适合处理分类目标变量
        - custom: 自定义分箱点，使用用户指定的分箱边界进行分箱
    
    Args:
        method: 分箱方法，默认为'equal_freq'。可选值：
               'equal_width', 'equal_freq', 'chi2', 'kmeans', 'mdlp'
        
        n_bins: 分箱数量，默认为10。
               - 在 equal_width, equal_freq, kmeans 方法中生效
               - 在 chi2, mdlp 方法中作为初始分箱数，最终分箱数可能会更少
               - 在使用 custom_bins 时被忽略
        
        custom_bins: 自定义分箱点列表，如果提供则忽略method和n_bins。
                    例如：[0, 18, 30, 50, float('inf')]
        
        min_samples: 每个分箱的最小样本占比，取值范围[0,1]，默认为0.05。
                    - 在 equal_freq, chi2, mdlp 方法中生效
                    - 在 equal_width, kmeans, custom 方法中不生效
                    例如：设置为0.05时，每个分箱至少要包含5%的样本
        
        max_bins: 最大分箱数，默认为50。
                 - 在 chi2, mdlp 方法中生效，用于限制最大分箱数
                 - 在其他方法中不生效
        
        chi_merge_threshold: 卡方分箱的合并阈值，默认为0.1。
                           - 仅在 chi2 方法中生效
                           - 值越大，越容易合并，最终分箱数越少
                           - 值越小，越不容易合并，最终分箱数越多
        
    Examples:
        >>> # 等频分箱，每个分箱至少包含10%的样本
        >>> binner = Binning(method='equal_freq', n_bins=5, min_samples=0.1)
        >>> binned = binner.fit_transform(data['age'])
        
        >>> # 自定义分箱
        >>> custom_bins = [0, 18, 30, 50, float('inf')]
        >>> binner = Binning(custom_bins=custom_bins)
        >>> binned = binner.fit_transform(data['age'])
        
        >>> # 卡方分箱，最大分箱数30，合并阈值0.1
        >>> binner = Binning(method='chi2', max_bins=30, chi_merge_threshold=0.1)
        >>> binned = binner.fit_transform(data['age'], data['target'])
        
        >>> # MDLP分箱，每个分箱至少包含5%的样本
        >>> binner = Binning(method='mdlp', min_samples=0.05)
        >>> binned = binner.fit_transform(data['age'], data['target'])
    """
    def __init__(
        self,
        method: str = 'equal_freq',
        n_bins: int = 10,
        custom_bins: Optional[List[float]] = None,
        min_samples: float = 0.05,
        max_bins: int = 50,
        chi_merge_threshold: float = 0.1,
    ):
        self.method = method
        self.n_bins = n_bins
        self.custom_bins = custom_bins
        self.min_samples = min_samples
        self.max_bins = max_bins
        self.chi_merge_threshold = chi_merge_threshold
        self.binning_result = None

    def _equal_width_binning(self, series: np.ndarray) -> List[float]:
        """等宽分箱
        
        将数据按照相等的区间宽度进行分箱
        
        Args:
            series: 输入数据数组
            
        Returns:
            分箱边界值列表
        """
        return list(np.linspace(series.min(), series.max(), self.n_bins + 1))

    def _equal_freq_binning(self, series: np.ndarray) -> List[float]:
        """等频分箱
        
        将数据按照每个箱中样本数量大致相等的原则进行分箱，
        同时确保每个分箱的样本量不少于指定的最小比例
        
        Args:
            series: 输入数据数组
            
        Returns:
            分箱边界值列表
        """
        min_samples_count = int(len(series) * self.min_samples)
        max_possible_bins = min(self.n_bins, len(series) // min_samples_count)
        actual_bins = max(2, max_possible_bins)  # 至少保留2个分箱
        
        quantiles = np.linspace(0, 1, actual_bins + 1)
        return list(np.quantile(series, quantiles))

    def _chi2_binning(self, values: np.ndarray, target: np.ndarray) -> List[float]:
        """卡方分箱
        
        基于卡方检验的自适应分箱方法，适合处理分类目标变量
        
        Args:
            values: 输入特征数组
            target: 目标变量数组
            
        Returns:
            分箱边界值列表
        """
        # 初始使用等频分箱
        initial_bins = self._equal_freq_binning(values)
        bins = initial_bins.copy()
        
        while len(bins) > 2:  # 至少保留两个分箱
            chi_values = []
            for i in range(1, len(bins)-1):
                temp_bins = bins.copy()
                del temp_bins[i]
                
                digitized = np.digitize(values, temp_bins)
                contingency = pd.crosstab(digitized, target)
                chi2_stat = stats.chi2_contingency(contingency)[0]
                chi_values.append(chi2_stat)
            
            min_chi_idx = np.argmin(chi_values) + 1
            if chi_values[min_chi_idx-1] > self.chi_merge_threshold:
                break
                
            del bins[min_chi_idx]
            
        return bins

    def _kmeans_binning(self, series: np.ndarray) -> List[float]:
        """KMeans分箱"""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
        kmeans.fit(series.reshape(-1, 1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        bins = [float('-inf')]
        bins.extend((centers[i] + centers[i+1])/2 for i in range(len(centers)-1))
        bins.append(float('inf'))
        return bins

    def _mdlp_binning(self, values: np.ndarray, target: np.ndarray) -> List[float]:
        """最小描述长度分箱(MDLP)"""
        def entropy(y):
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities))

        def find_best_split(x, y):
            best_gain = 0
            best_split = None
            
            for split in np.unique(x)[:-1]:
                left_mask = x <= split
                right_mask = ~left_mask
                
                n = len(y)
                n_left = sum(left_mask)
                n_right = sum(right_mask)
                
                gain = entropy(y) - (
                    n_left/n * entropy(y[left_mask]) +
                    n_right/n * entropy(y[right_mask])
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = split
                    
            return best_split, best_gain

        def recursive_split(x, y, bins):
            split, gain = find_best_split(x, y)
            if split is not None and gain > 0:
                bins.add(split)
                left_mask = x <= split
                recursive_split(x[left_mask], y[left_mask], bins)
                recursive_split(x[~left_mask], y[~left_mask], bins)

        bins = {float('-inf'), float('inf')}
        recursive_split(values, target, bins)
        return sorted(list(bins))

    def _calculate_bin_stats(self, values: np.ndarray, bins: List[float], target: Optional[np.ndarray] = None) -> Dict:
        """计算每个分箱的统计信息
        
        Args:
            values: 输入特征数组
            bins: 分箱边界值列表
            target: 目标变量数组（可选）
            
        Returns:
            包含每个分箱统计信息的字典，格式如：
            {
                '[0, 10)': {
                    'count': 100,
                    'min': 0.5,
                    'max': 9.8,
                    'mean': 5.2,
                    'target_mean': 0.3,  # 如果提供了target
                    'target_count': 30   # 如果提供了target
                },
                ...
            }
        """
        digitized = np.digitize(values, bins)
        stats_dict = {}
        
        for bin_idx in range(1, len(bins)):
            mask = digitized == bin_idx
            bin_label = f"[{bins[bin_idx-1]:.2f}, {bins[bin_idx]:.2f})"
            
            stats = {
                'count': sum(mask),
                'min': float(values[mask].min()) if any(mask) else None,
                'max': float(values[mask].max()) if any(mask) else None,
                'mean': float(values[mask].mean()) if any(mask) else None,
            }
            
            if target is not None:
                bin_target = target[mask]
                stats.update({
                    'target_mean': float(bin_target.mean()) if any(mask) else None,
                    'target_count': int(bin_target.sum()) if any(mask) else 0,
                })
            
            stats_dict[bin_label] = stats
            
        return stats_dict

    def fit(self, values: np.ndarray, target: Optional[np.ndarray] = None) -> 'Binning':
        """拟合分箱器
        
        Args:
            values: 输入特征数组
            target: 目标变量数组（可选，某些分箱方法需要）
            
        Returns:
            self，支持链式调用
            
        Raises:
            ValueError: 当使用需要目标变量的分箱方法但未提供target时
        """
        if self.custom_bins is not None:
            bins = self.custom_bins.copy()  # 使用copy避免修改原始数据
        else:
            if self.method == 'equal_width':
                bins = self._equal_width_binning(values)
            elif self.method in ['equal_freq', 'quantile']:
                bins = self._equal_freq_binning(values)
            elif self.method == 'chi2':
                if target is None:
                    raise ValueError("Target is required for chi-square binning")
                bins = self._chi2_binning(values, target)
            elif self.method == 'kmeans':
                bins = self._kmeans_binning(values)
            elif self.method == 'mdlp':
                if target is None:
                    raise ValueError("Target is required for MDLP binning")
                bins = self._mdlp_binning(values, target)
            else:
                raise ValueError(f"Unknown binning method: {self.method}")

            # 只有在非自定义分箱的情况下才处理无穷值
            if np.inf in bins or -np.inf in bins:
                min_val = float(values.min())
                max_val = float(values.max())
                bins = [b if b not in [np.inf, -np.inf] else (min_val-1 if b == -np.inf else max_val+1) for b in bins]

        # 计算分箱统计信息
        bin_stats = self._calculate_bin_stats(values, bins, target)
        
        # 生成分箱标签和编号映射
        bin_labels = []
        bin_indices = {}
        for i in range(len(bins)-1):
            if i == 0 and bins[i] == float('-inf'):
                label = f"(-inf, {bins[i+1]:.2f})"
            elif i == len(bins)-2 and bins[i+1] == float('inf'):
                label = f"[{bins[i]:.2f}, inf)"
            else:
                label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
            bin_labels.append(label)
            bin_indices[i] = label
        
        self.binning_result = BinningResult(
            bins=bins,
            bin_labels=bin_labels,
            bin_stats=bin_stats,
            bin_indices=bin_indices
        )
        
        return self

    def transform(self, values: np.ndarray, return_labels: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """将数据转换为分箱编号或标签
        
        Args:
            values: 输入特征数组
            return_labels: 是否同时返回分箱标签，默认False
            
        Returns:
            如果return_labels为False:
                返回分箱编号数组（从0开始的整数数组）
            如果return_labels为True:
                返回元组 (分箱编号数组, 分箱标签数组)
            
        Examples:
            >>> binner = Binning(method='equal_freq', n_bins=3)
            >>> binner.fit(data)
            
            # 只返回分箱编号
            >>> indices = binner.transform(data)  # array([0, 1, 2, 1, 0, ...])
            
            # 同时返回分箱编号和标签
            >>> indices, labels = binner.transform(data, return_labels=True)
            >>> indices  # array([0, 1, 2, 1, 0, ...])
            >>> labels   # array(['[0, 10)', '[10, 20)', '[20, inf)', ...])
        """
        if self.binning_result is None:
            raise ValueError("Binning not fitted yet. Call fit() first.")
            
        # 使用numpy的searchsorted来进行分箱
        indices = np.searchsorted(self.binning_result.bins, values, side='right') - 1
        
        # 处理边界情况
        indices = np.clip(indices, 0, len(self.binning_result.bin_labels) - 1)
        
        if return_labels:
            # 返回分箱标签
            return indices, np.array([self.binning_result.bin_labels[i] for i in indices])
        else:
            # 返回分箱编号
            return indices

    def fit_transform(self, values: np.ndarray, target: Optional[np.ndarray] = None, 
                     return_labels: bool = False) -> np.ndarray:
        """拟合分箱器并转换数据"""
        return self.fit(values, target).transform(values, return_labels)
    

"""
# 单独使用Binning类
binner = Binning(method='equal_freq', n_bins=10)
binned_values = binner.fit_transform(data['age'], data['target'])
"""