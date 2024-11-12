from dags.stage import CustomStage
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Liquid
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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
                    subtitle=f"基于 {len(self.feature_cols)} 个特征计算"
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
        
        self._create_similarity_chart(float(similarity))

        self.logger.info(result)