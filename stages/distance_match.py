import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Literal, Optional, Tuple, List
from dags.stage import CustomStage


class DistanceMatch(CustomStage):
    """距离匹配
    
    支持多种匹配方法：
    - nearest: 最近邻匹配
    - radius: 半径匹配
    - kernel: 核匹配
    - stratified: 分层匹配
    
    Args:
        cols: 用于匹配的特征列
        proba_col: 倾向性得分列
        method: 匹配方法
        caliper: 匹配阈值
        k: 最近邻个数 (k=1为1:1匹配)
        kernel: 核函数类型
        n_strata: 分层数量
    """
    
    def __init__(
        self, 
        cols: list,
        proba_col: str,
        id_col: str = None,
        method: Literal['nearest', 'radius', 'kernel', 'stratified'] = 'nearest',
        caliper: float = 0.2,
        k: Optional[int] = None,
        kernel: Literal['gaussian', 'epanechnikov'] = 'gaussian',
        n_strata: int = 5,
        need_normalize: bool = True
    ):
        super().__init__(n_outputs=1)
        self.cols = cols
        self.proba_col = proba_col
        self.id_col = id_col
        self.method = method
        self.caliper = caliper
        # 为不同方法设置默认的k值
        self.k = k if k is not None else (1 if method in ['nearest', 'kernel', 'stratified'] else None)
        self.kernel = kernel
        self.n_strata = n_strata
        self.need_normalize = need_normalize
        self.scaler = StandardScaler() if need_normalize else None
            
    def nearest_neighbor_match(
        self, 
        X_A: np.ndarray, 
        X_B: np.ndarray, 
        ids_A: np.ndarray,
        ids_B: np.ndarray
    ) -> List[Tuple]:
        """最近邻匹配（排序版本）"""
        # 计算距离矩阵
        distances = cdist(X_A, X_B)
        
        # 获取每行（每个A样本）的排序索引
        sorted_indices = np.argsort(distances, axis=1)
        
        matched_pairs = []
        used_B = np.zeros(len(X_B), dtype=bool)
        
        # 使用numpy的布尔索引加速匹配过程
        for i in range(len(X_A)):
            # 获取当前A样本的所有候选B样本
            candidates = sorted_indices[i]
            
            # 找出未使用且距离在阈值内的B样本
            valid_candidates = candidates[
                (~used_B[candidates]) & 
                (distances[i, candidates] <= self.caliper)
            ]
            
            # 如果找到足够的匹配样本
            if len(valid_candidates) >= self.k:
                for j in valid_candidates[:self.k]:
                    matched_pairs.append((ids_A[i], ids_B[j]))
                    used_B[j] = True
                
            if used_B.sum() + self.k > len(X_B):
                break
        
        return matched_pairs
    
    def radius_match(
        self,
        X_A: np.ndarray,
        X_B: np.ndarray,
        ids_A: np.ndarray,
        ids_B: np.ndarray
    ) -> List[Tuple]:
        """半径匹配"""
        distances = cdist(X_A, X_B)
        matched_pairs = []
        
        # 找出所有在半径范围内的匹配
        for i in range(len(X_A)):
            matches = np.where(distances[i] <= self.caliper)[0]
            for j in matches:
                matched_pairs.append((ids_A[i], ids_B[j]))
                
        return matched_pairs
    
    def kernel_match(
        self,
        X_A: np.ndarray,
        X_B: np.ndarray,
        ids_A: np.ndarray,
        ids_B: np.ndarray
    ) -> List[Tuple]:
        """核匹配
        
        使用核函数计算权重，为每个处理组样本选择最相似的对照组样本
        """
        distances = cdist(X_A, X_B)
        matched_pairs = []
        used_B = set()
        
        # 如果没有指定k，默认为1
        k = self.k if self.k is not None else 1
        
        # 计算核权重
        if self.kernel == 'gaussian':
            weights = np.exp(-distances**2 / (2 * self.caliper**2))
        else:  # epanechnikov
            weights = np.maximum(0, 1 - (distances/self.caliper)**2)
        
        # 为每个A样本找到权重最大的k个B样本
        for i in range(len(X_A)):
            # 获取未使用的B样本的权重
            available_weights = [(j, w) for j, w in enumerate(weights[i]) 
                               if j not in used_B and w > 0]
            
            # 按权重排序
            available_weights.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前k个权重最大的样本
            for j, w in available_weights[:k]:
                matched_pairs.append((ids_A[i], ids_B[j]))
                used_B.add(j)
                
                # 如果已经没有足够的B样本可匹配
                if len(used_B) + k > len(X_B):
                    break
                
            # 如果已经没有足够的B样本可匹配
            if len(used_B) + k > len(X_B):
                break
        
        return matched_pairs
    
    def stratified_match(
        self,
        X_A: np.ndarray,
        X_B: np.ndarray,
        ids_A: np.ndarray,
        ids_B: np.ndarray
    ) -> List[Tuple]:
        """分层匹配"""
        # 使用倾向性得分进行分层
        scores = np.concatenate([X_A[:, 0], X_B[:, 0]])  # 假设第一列是倾向性得分
        bins = np.percentile(scores, np.linspace(0, 100, self.n_strata + 1))
        
        matched_pairs = []
        
        # 在每一层内进行最近邻匹配
        for i in range(len(bins) - 1):
            mask_A = (X_A[:, 0] >= bins[i]) & (X_A[:, 0] < bins[i + 1])
            mask_B = (X_B[:, 0] >= bins[i]) & (X_B[:, 0] < bins[i + 1])
            
            if np.any(mask_A) and np.any(mask_B):
                # 创建一个临时的匹配器用于当前层
                temp_matcher = DistanceMatch(
                    cols=self.cols,
                    proba_col=self.proba_col,
                    id_col=self.id_col,
                    method='nearest',
                    caliper=self.caliper,
                    k=self.k,
                    need_normalize=False  # 已经标准化过了
                )
                
                strata_pairs = temp_matcher.nearest_neighbor_match(
                    X_A[mask_A], 
                    X_B[mask_B],
                    ids_A[mask_A],
                    ids_B[mask_B]
                )
                matched_pairs.extend(strata_pairs)
                
        return matched_pairs

    def forward(self, lf_A: pl.LazyFrame, lf_B: pl.LazyFrame) -> pl.LazyFrame:
        """执行匹配过程
        
        Args:
            lf_A: 处理组数据
            lf_B: 对照组数据（数据量可能与A不同）
        """
        # 收集数据并生成唯一ID
        if self.id_col:
            df_A = (
                lf_A.select(
                    [pl.col(self.id_col)] + 
                    [pl.col(self.proba_col)] + 
                    [pl.col(c) for c in self.cols]
                )
                .collect()
            )
            df_B = (
                lf_B.select(
                    [pl.col(self.id_col)] + 
                    [pl.col(self.proba_col)] + 
                    [pl.col(c) for c in self.cols]
                )
                .collect()
            )
        else:
            df_A = (
                lf_A.select(
                    [pl.col(self.proba_col)] + 
                    [pl.col(c) for c in self.cols]
                )
                .with_row_count("id")  # 使用 with_row_count 替代 np.arange
                .collect()
            )
            df_B = (
                lf_B.select(
                    [pl.col(self.proba_col)] + 
                    [pl.col(c) for c in self.cols]
                )
                .with_row_count("id")
                .collect()
            )
        
        # 检查数据有效性
        if len(df_A) == 0 or len(df_B) == 0:
            self.logger.warning("存在空数据集")
            return pl.DataFrame({"id_A": [], "id_B": []}).lazy()
        
        # 检查是否有缺失值
        null_count_A = df_A.null_count().sum_horizontal().item()
        null_count_B = df_B.null_count().sum_horizontal().item()
        if null_count_A > 0 or null_count_B > 0:
            # 可以选择去除缺失值或抛出异常
            df_A = df_A.drop_nulls()
            df_B = df_B.drop_nulls()
            self.logger.info(f"已去除缺失值, 去除数量: A组={null_count_A}, B组={null_count_B}")
        
        # 转换为numpy数组
        X_A = df_A.select([self.proba_col] + self.cols).to_numpy()
        X_B = df_B.select([self.proba_col] + self.cols).to_numpy()
        if self.id_col:
            ids_A = df_A[self.id_col].to_numpy()
            ids_B = df_B[self.id_col].to_numpy()
        else:
            ids_A = df_A['id'].to_numpy()
            ids_B = df_B['id'].to_numpy()
        
        # 标准化特征
        if self.need_normalize:
            X = np.vstack([X_A, X_B])
            X = self.scaler.fit_transform(X)
            X_A, X_B = X[:len(X_A)], X[len(X_A):]
        
        # 根据方法选择匹配策略
        match_methods = {
            'nearest': self.nearest_neighbor_match,
            'radius': self.radius_match,
            'kernel': self.kernel_match,
            'stratified': self.stratified_match
        }
        
        matched_pairs = match_methods[self.method](X_A, X_B, ids_A, ids_B)
        
        # 记录详细的匹配结果
        self.logger.info(f"匹配方法: {self.method}")
        self.logger.info(f"A组原始样本数: {lf_A.select(pl.len()).collect().item()}")
        self.logger.info(f"B组原始样本数: {lf_B.select(pl.len()).collect().item()}")
        self.logger.info(f"A组有效样本数: {len(X_A)}")
        self.logger.info(f"B组有效样本数: {len(X_B)}")
        self.logger.info(f"匹配对数: {len(matched_pairs)}")
        self.logger.info(f"A组匹配率: {len(matched_pairs)/len(X_A):.2%}")
        self.logger.info(f"B组匹配率: {len(matched_pairs)/len(X_B):.2%}")
        
        # 构建匹配结果DataFrame
        matched_df = pl.DataFrame({
            'id_A_matched': [p[0] for p in matched_pairs],
            'id_B_matched': [p[1] for p in matched_pairs]
        })
        
        return matched_df.lazy()
