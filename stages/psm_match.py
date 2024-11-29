import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Literal, Optional, Tuple, List
from dags.stage import CustomStage
from stages.utils.plot_img import plot_proba_distribution, plot_hist_compare
from pyecharts.charts import Grid
from pyecharts import options as opts
from scipy import stats


class PSMMatch(CustomStage):
    """PSM匹配
    
    支持多种匹配方法：
    - nearest: 最近邻匹配
    - radius: 半径匹配
    - kernel: 核匹配
    - stratified: 分层匹配
    
    Args:
        proba_col: 倾向性得分列
        method: 匹配方法
        caliper: 匹配阈值
        k: 最近邻个数 (k=1为1:1匹配)
        kernel: 核函数类型
        n_strata: 分层数量
    """
    
    def __init__(
        self, 
        proba_col: str,
        id_col: str = None,
        analysis_cols: List[str] = [],
        method: Literal['nearest', 'radius', 'kernel', 'stratified'] = 'nearest',
        caliper: float = 0.2,
        k: Optional[int] = None,
        kernel: Literal['gaussian', 'epanechnikov'] = 'gaussian',
        n_strata: int = 5,
    ):
        super().__init__(n_outputs=1)
        self.proba_col = proba_col
        self.id_col = id_col
        self.method = method
        self.caliper = caliper
        self.k = k if k is not None else (1 if method in ['nearest', 'kernel', 'stratified'] else None)
        self.kernel = kernel
        self.n_strata = n_strata
        self.analysis_cols = analysis_cols
        # TODO 倾向得分阈值匹配
            
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
        matched_A_idx = []
        matched_B_idx = []
        used_B = np.zeros(len(X_B), dtype=bool)
        
        # 修改匹配逻辑
        for i in range(len(X_A)):
            candidates = sorted_indices[i]
            
            # 找出未使用且距离在阈值内的B样本
            valid_candidates = candidates[
                (~used_B[candidates]) & 
                (distances[i, candidates] <= self.caliper)
            ]
            
            # 只选择一个最近的匹配样本
            if len(valid_candidates) > 0:  # 只要有有效候选即可
                j = valid_candidates[0]  # 取第一个（最近的）候选
                matched_pairs.append((ids_A[i], ids_B[j]))
                matched_A_idx.append(i)
                matched_B_idx.append(j)
                used_B[j] = True
                
            if used_B.sum() >= len(X_B):
                break

        return matched_pairs, matched_A_idx, matched_B_idx
    
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
        matched_A_idx = []
        matched_B_idx = []
        used_B = np.zeros(len(X_B), dtype=bool)
        
        # 找出所有在半径范围内的匹配
        for i in range(len(X_A)):
            matches = np.where((distances[i] <= self.caliper) & (~used_B))[0]
            for j in matches:
                matched_pairs.append((ids_A[i], ids_B[j]))
                matched_A_idx.append(i)
                matched_B_idx.append(j)
                used_B[j] = True
                
        return matched_pairs, matched_A_idx, matched_B_idx
    
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
        matched_A_idx = []
        matched_B_idx = []
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
                matched_A_idx.append(i)
                matched_B_idx.append(j)
                used_B.add(j)
                
                # 如果已经没有足够的B样本可匹配
                if len(used_B) + k > len(X_B):
                    break
                
            # 如果已经没有足够的B样本可匹配
            if len(used_B) + k > len(X_B):
                break
        
        return matched_pairs, matched_A_idx, matched_B_idx
    
    def stratified_match(self, X_A: np.ndarray, X_B: np.ndarray, ids_A: np.ndarray, ids_B: np.ndarray) -> List[Tuple]:
        """分层匹配"""
        # 直接使用倾向性得分进行分层
        scores = np.concatenate([X_A.ravel(), X_B.ravel()])
        bins = np.percentile(scores, np.linspace(0, 100, self.n_strata + 1))
        
        matched_pairs = []
        matched_A_idx = []
        matched_B_idx = []
        
        # 在每一层内进行最近邻匹配
        for i in range(len(bins) - 1):
            mask_A = (X_A.ravel() >= bins[i]) & (X_A.ravel() < bins[i + 1])
            mask_B = (X_B.ravel() >= bins[i]) & (X_B.ravel() < bins[i + 1])
            
            if np.any(mask_A) and np.any(mask_B):
                # 创建一个临时的匹配器用于当前层
                temp_matcher = PSMMatch(
                    proba_col=self.proba_col,
                    id_col=self.id_col,
                    method='nearest',
                    caliper=self.caliper,
                    k=self.k
                )
                
                strata_pairs, strata_matched_A_idx, strata_matched_B_idx = temp_matcher.nearest_neighbor_match(
                    X_A[mask_A], 
                    X_B[mask_B],
                    ids_A[mask_A],
                    ids_B[mask_B]
                )
                matched_pairs.extend(strata_pairs)
                matched_A_idx.extend(strata_matched_A_idx)  
                matched_B_idx.extend(strata_matched_B_idx)
                
        return matched_pairs, matched_A_idx, matched_B_idx

    def calculate_smd(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """计算标准化均值差异(SMD)
        
        SMD = |mean1 - mean2| / sqrt((var1 + var2) / 2)
        
        Args:
            x1: 第一组数据
            x2: 第二组数据
        
        Returns:
            float: SMD值
        """
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1, var2 = np.var(x1), np.var(x2)
        
        # 处理分母为0的情况
        pooled_var = (var1 + var2) / 2
        if pooled_var == 0:
            return 0
        
        smd = np.abs(mean1 - mean2) / np.sqrt(pooled_var)
        return smd

    def _evaluate_smd(self, smd: float) -> str:
        """评估SMD
        
        Args:
            smd: SMD值
        
        Returns:
            str: 评估结果
        """
        if smd < 0.1:
            return "很好"
        elif smd < 0.2:
            return "一般"
        else:
            return "较差"

    def forward(self, lf_A: pl.LazyFrame, lf_B: pl.LazyFrame) -> pl.LazyFrame:
        """执行匹配过程
        
        Args:
            lf_A: 处理组数据
            lf_B: 对照组数据（数据量可能与A不同）
        """

        # 收集数据并生成唯一ID
        # if self.id_col:
        #     lf_A = lf_A.select([pl.col(self.id_col), pl.col(self.proba_col)]).collect()
        #     lf_B = lf_B.select([pl.col(self.id_col), pl.col(self.proba_col)]).collect()
        # else:
        #     lf_A = (
        #         lf_A.select(pl.col(self.proba_col))
        #         .with_row_count("id")
        #         .collect()
        #     )
        #     lf_B = (
        #         lf_B.select(pl.col(self.proba_col))
        #         .with_row_count("id")
        #         .collect()
        #     )
        if self.id_col not in lf_A.columns:
            lf_A = lf_A.with_row_count("id")
        if self.id_col not in lf_B.columns:
            lf_B = lf_B.with_row_count("id")

        proba_A = lf_A.select(pl.col(self.proba_col)).collect().to_numpy()
        proba_B = lf_B.select(pl.col(self.proba_col)).collect().to_numpy()

        self.logger.info(f"实验组样本数: {len(proba_A)}, 对照组样本数: {len(proba_B)}")
        self.logger.info(f"实验组倾向得分范围: {proba_A.min()} ~ {proba_A.max()}")
        self.logger.info(f"对照组倾向得分范围: {proba_B.min()} ~ {proba_B.max()}")

        if self.id_col:
            ids_A = lf_A.select(pl.col(self.id_col)).collect().to_numpy()
            ids_B = lf_B.select(pl.col(self.id_col)).collect().to_numpy()
        else:
            ids_A = lf_A.select(pl.col("id")).collect().to_numpy()
            ids_B = lf_B.select(pl.col("id")).collect().to_numpy()
        
        # 根据方法选择匹配策略
        match_methods = {
            'nearest': self.nearest_neighbor_match,
            'radius': self.radius_match,
            'kernel': self.kernel_match,
            'stratified': self.stratified_match
        }
        
        matched_pairs, matched_A_idx, matched_B_idx = match_methods[self.method](proba_A, proba_B, ids_A, ids_B)

        self.logger.warning("""Tips: 
        p > 0.1匹配效果很好
        0.05 < p < 0.1：匹配效果可以接受
        0.01 < p < 0.05：匹配效果一般
        p < 0.01：匹配效果差""")

        # 在这里把实验组和对照组的匹配前后所有特征的分布绘制出来、进行检验。
        for col in self.analysis_cols:
            if col == self.proba_col or col == self.id_col:
                self.logger.info(f"跳过列: {col}")
                continue

            # 获取匹配前数据
            A_feature_data = lf_A.select(pl.col(col)).collect().to_numpy().flatten()
            B_feature_data = lf_B.select(pl.col(col)).collect().to_numpy().flatten()
            
            # 获取匹配后数据
            A_matched_data = A_feature_data[matched_A_idx]
            B_matched_data = B_feature_data[matched_B_idx]
            
            # 判断是否为连续型变量（通过唯一值数量判断）
            is_continuous = len(np.unique(A_feature_data)) > 10
            
            if is_continuous:
                # Z检验（对连续型变量）
                after_stat, after_pvalue = stats.ranksums(A_matched_data, B_matched_data)
                
                self.logger.info(f"\n特征 {col} (连续型) 的检验结果:")
                self.logger.info(f"匹配后 - Z统计量: {after_stat:.4f}, p值: {after_pvalue:.4f}")
                
            else:
                # 卡方检验（对离散型变量）
                try:
                    # 计算联列表
                    after_crosstab = np.histogram2d(
                        A_matched_data, B_matched_data,
                        bins=[np.unique(A_matched_data), np.unique(B_matched_data)]
                    )[0]
                    
                    if after_crosstab.size > 0:
                        after_chi2, after_pvalue = stats.chi2_contingency(after_crosstab)[:2]
                        self.logger.info(f"\n特征 {col} (离散型) 的检验结果:")
                        self.logger.info(f"匹配后 - 卡方统计量: {after_chi2:.4f}, p值: {after_pvalue:.4f}")
                    else:
                        self.logger.warning(f"\n特征 {col} (离散型) 无法进行卡方检验: 数据不足")
                except Exception as e:
                    self.logger.warning(f"\n特征 {col} (离散型) 进行卡方检验时出错: {str(e)}")
            
            # 计算匹配前后的SMD
            before_smd = self.calculate_smd(A_feature_data, B_feature_data)
            after_smd = self.calculate_smd(A_matched_data, B_matched_data)
            
            self.logger.info(f"\n特征 {col} 的SMD分析结果:")
            self.logger.info(f"匹配前 - SMD: {before_smd:.4f} ({self._evaluate_smd(before_smd)}) 匹配后 - SMD: {after_smd:.4f} ({self._evaluate_smd(after_smd)})")

            # 计算SMD改善程度
            smd_improvement = ((before_smd - after_smd) / before_smd * 100 
                              if before_smd != 0 else 0)
            self.logger.info(f"SMD改善程度: {smd_improvement:.2f}%")
            
            # 绘制分布图的代码保持不变
            before_match_chart = plot_hist_compare(A_feature_data, B_feature_data, title=f"")
            after_match_chart = plot_hist_compare(A_matched_data, B_matched_data, title=f"")
            
            # Chart绘制图表
            grid = (
                Grid()
                .add(before_match_chart, grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%"))
                .add(after_match_chart, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%"))
            )
            self.summary.append({f"{col}匹配前后分布对比": grid.dump_options_with_quotes()})
        
        # 记录详细的匹配结果
        self.logger.info(f"匹配方法: {self.method}")
        self.logger.info(f"匹配对数: {len(matched_pairs)}")
        self.logger.info(f"A组匹配率: {len(matched_pairs)/len(proba_A):.2%}")
        self.logger.info(f"B组匹配率: {len(matched_pairs)/len(proba_B):.2%}")
        
        # 构建匹配结果DataFrame
        matched_df = pl.DataFrame({
            'id_实验组_matched': [p[0][0] for p in matched_pairs],
            'id_对照组_matched': [p[1][0] for p in matched_pairs],
            '实验组-倾向得分': [float(proba_A[matched_A_idx[i]][0]) for i in range(len(matched_pairs))],
            '对照组-倾向得分': [float(proba_B[matched_B_idx[i]][0]) for i in range(len(matched_pairs))]
        })
        
        return matched_df.lazy()
