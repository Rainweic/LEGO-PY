from dags.stage import CustomStage
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Liquid
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Literal, Tuple
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats


class CustomerSimilarityStage(CustomStage):
    """使用MinHash LSH计算客户群体相似度"""
    
    def __init__(self, feature_cols=[]):
        super().__init__(n_outputs=0)
        self.cols = feature_cols
        self.scaler = StandardScaler()
        
    def _create_similarity_chart(self, similarity: float) -> dict:
        """创建相似度水滴图
        
        Args:
            similarity: 相似度值 (0-1)
            
        Returns:
            包含图表配置的字典
        """
        # 创建水滴图
        liquid = (
            Liquid()
            .add(
                "相似度",
                [similarity],  # 数据值
                label_opts=opts.LabelOpts(
                    font_size=50,
                    formatter=lambda x: f"{x * 100:.1f}%",
                    position="inside"
                ),
                is_outline_show=False,
                shape='circle',  # 可选 'circle', 'rect', 'roundRect', 'triangle', 'diamond', 'pin'
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="客户群体相似度",
                    subtitle=f"基于 {len(self.cols)} 个特征计算"
                )
            )
        )
        
        # 定义式
        liquid.options.get('series')[0].update(
            itemStyle_opts=opts.ItemStyleOpts(
                opacity=0.8,
            ),
            outline_border_distance=2,
            outline_item_style_opts=opts.ItemStyleOpts(
                border_color="#294D99",
                border_width=2
            ),
            color=[
                "rgb(41, 77, 153)",
                "rgb(51, 97, 173)",
                "rgb(61, 117, 193)"
            ],
            background_color="rgba(255, 255, 255, 0.8)"
        )
        
        return {"相似度": liquid.dump_options_with_quotes()}

    def _detect_feature_types(self, df: pl.LazyFrame):
        """自动检测特征类型"""
        df_schema = df.collect_schema()
        numerical_features = []
        categorical_features = []
        
        for col in self.cols:  # 使用初始化时传入的特征列
            dtype = df_schema[col]
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                numerical_features.append(col)
            else:
                categorical_features.append(col)
                
        return numerical_features, categorical_features
    
    def forward(self, group1_df: pl.LazyFrame, group2_df: pl.LazyFrame):
        """计算两个客户群的相似度
        
        Args:
            group1_df: 第一个客户群的LazyFrame
            group2_df: 第二个客户群的LazyFrame
        """
        if not self.cols:
            raise ValueError("必须指定特征列！")
            
        # 1. 自动检测特征类型
        numerical_features, categorical_features = self._detect_feature_types(group1_df)
        
        features1 = []
        features2 = []
        
        # 2. 处理数值特征
        if numerical_features:
            numerical_data1 = group1_df.select(numerical_features).collect().to_numpy()
            numerical_data2 = group2_df.select(numerical_features).collect().to_numpy()
            
            # 标准化
            numerical_data1 = self.scaler.fit_transform(numerical_data1)
            numerical_data2 = self.scaler.transform(numerical_data2)
            
            features1.append(numerical_data1)
            features2.append(numerical_data2)
        
        # 3. 处理类别特征
        if categorical_features:
            # 对每个类别特征进行编码
            for col in categorical_features:
                # 获取所有唯一值
                unique_values = (
                    pl.concat([
                        group1_df.select(col).collect(),
                        group2_df.select(col).collect()
                    ])
                    .unique()
                    .to_series()
                    .to_list()
                )
                
                # 创建编码器
                encoder = LabelEncoder()
                encoder.fit(unique_values)
                
                # 编码两个组的数据
                cat_data1 = encoder.transform(
                    group1_df.select(col).collect().to_series().to_list()
                ).reshape(-1, 1)
                cat_data2 = encoder.transform(
                    group2_df.select(col).collect().to_series().to_list()
                ).reshape(-1, 1)
                
                features1.append(cat_data1)
                features2.append(cat_data2)
        
        # 4. 合并所有特征
        if not features1 or not features2:
            raise ValueError("没有有效的特征可以用于计算相似度！")
            
        features1 = np.hstack(features1)
        features2 = np.hstack(features2)
        
        # 5. 计算群体中心点
        centroid1 = np.mean(features1, axis=0).reshape(1, -1)  # 确保是2D数组
        centroid2 = np.mean(features2, axis=0).reshape(1, -1)  # 确保是2D数组
        
        # 6. 计算相似度
        cosine_sim = cosine_similarity(centroid1, centroid2)[0][0]
        euclidean_dist = euclidean_distances(centroid1, centroid2)[0][0]
        
        # 7. 综合相似度（归一化欧氏距离和余弦相似度的平均值）
        similarity = (1 / (1 + euclidean_dist) + cosine_sim) / 2
        
        # 8. 创建结果字典
        result = {
            'similarity_score': float(similarity),
            'details': {
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclidean_dist),
                'group1_size': len(features1),
                'group2_size': len(features2),
                'feature_count': features1.shape[1],
                'numerical_features': numerical_features,
                'categorical_features': categorical_features
            }
        }
        
        self.summary.append(self._create_similarity_chart(float(similarity)))
        self.logger.info(result)
        return result


class BinnedKLSimilarityStage(CustomerSimilarityStage):
    """基于分箱和KL散度的客户群体相似度计算"""
    
    def __init__(
        self, 
        feature_cols=[], 
        bin_method: Literal['equal_width', 'equal_freq', 'kmeans'] = 'equal_freq',
        n_bins: int = 10,
        smooth_factor: float = 1e-10,  # 平滑因子，避免出现0概率
        handle_outliers: bool = False,  # 是否处理异常值
    ):
        super().__init__(feature_cols)
        self.bin_method = bin_method
        self.n_bins = n_bins
        self.smooth_factor = smooth_factor if isinstance(smooth_factor, float) else eval(smooth_factor)
        self.handle_outliers = handle_outliers
        
    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        return np.sum(p * np.log(p / q))
    
    def _calculate_similarity(self, kl_div: float) -> float:
        """将KL散度转换为相似度分数(0-1)"""
        return 1 / (1 + kl_div)
    
    def _choose_bin_method(self, data: np.ndarray) -> str:
        # 检查数据分布
        skewness = stats.skew(data)
        if abs(skewness) > 2:
            return 'equal_freq'  # 偏态分布用等频分箱
        else:
            return 'equal_width'  # 正态分布用等宽分箱
    
    def _preprocess_feature(self, feat: np.ndarray) -> np.ndarray:
        """处理异常值"""
        q1, q3 = np.percentile(feat, [25, 75])  # 第一四分位数(25%分位点) 第三四分位数(75%分位点)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr  # 下界 = Q1 - 1.5倍IQR
        upper = q3 + 1.5 * iqr  # 上界 = Q3 + 1.5倍IQR
        return np.clip(feat, lower, upper)
    
    def forward(self, group1_df: pl.LazyFrame, group2_df: pl.LazyFrame):
        """计算两个客户群的相似度"""
        if not self.cols:
            raise ValueError("必须指定特征列！")
            
        # 直接获取所有特征数据
        data1 = group1_df.select(self.cols).collect()
        data2 = group2_df.select(self.cols).collect()
        
        feature_kl_divs = []
        feature_details = []
        
        for i, feature in enumerate(self.cols):
            feat1 = data1.get_column(feature)
            feat2 = data2.get_column(feature)
            
            # 对数值特征进行异常值处理
            if feat1.dtype not in [pl.Categorical, pl.Utf8] and self.handle_outliers:
                feat1 = self._preprocess_feature(feat1.to_numpy())
                feat2 = self._preprocess_feature(feat2.to_numpy())
            
            # 对于类别特征
            if feat1.dtype in [pl.Categorical, pl.Utf8]:
                # 1. 只用group1的数据确定主要类别
                categories = feat1.unique()
                
                # 2. 如果类别数大于n_bins或group2中有新类别，需要进行合并
                group2_categories = feat2.unique()
                new_categories_exist = any(cat not in categories for cat in group2_categories)
                
                if len(categories) > self.n_bins or new_categories_exist:
                    # 计算group1中各类别的频率
                    cat_freq = np.array([
                        (feat1 == cat).sum() / len(feat1)
                        for cat in categories
                    ])
                    # 保留频率最高的n_bins-1个类别
                    top_n = min(self.n_bins - 1, len(categories) - 1)
                    top_cats = categories[np.argsort(cat_freq)[-top_n:]]
                    bins = top_cats.to_list() + ['其他']
                else:
                    bins = categories.to_list()
                    top_cats = categories
                
                # 3. 计算两组数据在这些类别上的分布
                hist1 = np.zeros(len(bins))
                hist2 = np.zeros(len(bins))
                
                # 对group1计算分布
                for i, cat in enumerate(bins[:-1] if '其他' in bins else bins):
                    hist1[i] = (feat1 == cat).sum() / len(feat1)
                # 其他类别的频率
                if '其他' in bins:
                    hist1[-1] = sum((feat1 == cat).sum() for cat in categories if cat not in bins[:-1]) / len(feat1)
                
                # 对group2计算分布(新类别归入其他)
                for i, cat in enumerate(bins[:-1] if '其他' in bins else bins):
                    hist2[i] = (feat2 == cat).sum() / len(feat2)
                # 其他类别的频率(包括group2特有的类别)
                if '其他' in bins:
                    hist2[-1] = sum((feat2 == cat).sum() for cat in group2_categories if cat not in bins[:-1]) / len(feat2)
                
            else:  # 数值特征
                # 1. 先用group1的数据确定分箱边界
                data = feat1.to_numpy()
                if self.bin_method == 'equal_width':
                    bins = np.linspace(
                        data.min(), 
                        data.max(), 
                        self.n_bins + 1
                    )
                elif self.bin_method == 'equal_freq':
                    bins = np.percentile(
                        data,
                        np.linspace(0, 100, self.n_bins + 1)
                    )
                elif self.bin_method == 'kmeans':
                    kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
                    kmeans.fit(data.reshape(-1, 1))
                    centers = np.sort(kmeans.cluster_centers_.flatten())
                    bins = np.concatenate([
                        [data.min()],
                        (centers[:-1] + centers[1:]) / 2,
                        [data.max()]
                    ])
                
                # 2. 用相同的分箱边界计算两组数据的分布
                hist1, _ = np.histogram(feat1.to_numpy(), bins=bins, density=True)
                hist2, _ = np.histogram(feat2.to_numpy(), bins=bins, density=True)
            
            # 添加平滑因子并归一化
            hist1 = hist1 + self.smooth_factor
            hist2 = hist2 + self.smooth_factor
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # 计算KL散度并存储结果
            kl_div_12 = self._calculate_kl_divergence(hist1, hist2)
            kl_div_21 = self._calculate_kl_divergence(hist2, hist1)
            avg_kl_div = (kl_div_12 + kl_div_21) / 2
            
            feature_kl_divs.append(avg_kl_div)
            feature_details.append({
                'feature': feature,
                'kl_divergence': float(avg_kl_div),
                'bins': bins,
                'dist1': hist1.tolist(),
                'dist2': hist2.tolist()
            })
        
        # 计算总体相似度
        total_kl_div = stats.hmean([1 + kl for kl in feature_kl_divs]) - 1
        similarity = self._calculate_similarity(total_kl_div)
        
        # 创建结果字典
        result = {
            'similarity_score': float(similarity),
            'details': {
                'bin_method': self.bin_method,
                'n_bins': self.n_bins,
                'feature_details': feature_details,
                'group1_size': len(data1),
                'group2_size': len(data2),
                'feature_count': len(self.cols)
            }
        }
        
        self.summary.append(self._create_similarity_chart(float(similarity)))
        self.logger.info(result)
        return result