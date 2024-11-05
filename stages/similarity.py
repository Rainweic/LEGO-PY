from dags.stage import CustomStage
from datasketch import MinHash
from typing import Set, Dict, List, Tuple
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import xxhash
from pyecharts import options as opts
from pyecharts.charts import Liquid


class CustomerSimilarityStage(CustomStage):
    """计算两个客户群体的相似度的优化版本"""
    
    def __init__(
        self, 
        feature_cols: list[str], 
        num_perm: int = 128,
        sample_size: int = None,  # 采样大小
        n_threads: int = 4,       # 并行线程数
        cache_size: int = 128     # LRU缓存大小
    ):
        super().__init__(n_outputs=1)
        self.feature_cols = feature_cols
        self.num_perm = num_perm
        self.sample_size = sample_size
        self.n_threads = n_threads
        self.cache_size = cache_size
        
    @staticmethod
    def _fast_hash(value: str) -> int:
        """使用xxhash进行快速哈希计算"""
        return xxhash.xxh64(value).intdigest()
    
    def _sample_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """智能采样数据
        
        如果数据量超过sample_size，进行分层采样以保持数据分布
        """
        if not self.sample_size:
            return df
            
        total_rows = df.select(pl.len()).collect().item()
        if total_rows <= self.sample_size:
            return df
            
        # 计算每个特征组合的采样比例
        sample_fraction = self.sample_size / total_rows
        
        # LazyFrame不支持sample方法，需先collect再采样
        collected_df = df.collect()
        sampled_df = (
            collected_df.group_by(self.feature_cols)
            .agg(pl.col("*"))
            .sample(fraction=sample_fraction, seed=42)
        )
        
        return sampled_df.lazy()  # 返回LazyFrame
    
    @lru_cache(maxsize=128)
    def _get_feature_hash(self, feature_str: str) -> int:
        """缓存特征哈希值"""
        return self._fast_hash(feature_str)
    
    def _process_column_batch(
        self, 
        df: pl.DataFrame, 
        cols: List[str]
    ) -> Set[int]:
        """并行处理一批特征列"""
        features = set()
        for col in cols:
            col_values = df.get_column(col).unique().to_list()
            features.update(
                self._get_feature_hash(f"{col}_{val}") 
                for val in col_values
            )
        return features
    
    def _create_feature_set(self, df: pl.LazyFrame) -> Set[int]:
        """优化的特征集合创建
        
        1. 数据采样
        2. 并行处理
        3. 哈希缓存
        """
        # 采样
        sampled_df = self._sample_data(df)
        
        # 收集数据
        df_collected = sampled_df.collect()
        
        # 将特征列分成多个批次
        batch_size = max(1, len(self.feature_cols) // self.n_threads)
        column_batches = [
            self.feature_cols[i:i + batch_size]
            for i in range(0, len(self.feature_cols), batch_size)
        ]
        
        # 并行处理每个批次
        features = set()
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_column_batch, 
                    df_collected, 
                    batch
                ): batch 
                for batch in column_batches
            }
            
            for future in future_to_batch:
                features.update(future.result())
                
        return features
    
    def _create_minhash(self, features: Set[int]) -> MinHash:
        """创建MinHash对象"""
        minhash = MinHash(num_perm=self.num_perm)
        for feature in features:
            # 直接使用整数特征值
            minhash.update(str(feature).encode('utf8'))
        return minhash
    
    def _calculate_similarity(
        self, 
        set1: Set[int], 
        set2: Set[int]
    ) -> Tuple[float, Dict]:
        """使用MinHash计算相似度"""
        # 创建MinHash
        minhash1 = self._create_minhash(set1)
        minhash2 = self._create_minhash(set2)
        
        # 计算MinHash估计的Jaccard相似度
        similarity = minhash1.jaccard(minhash2)
        
        # 计算实际集合的大小(用于详细信息)
        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))
        
        details = {
            "intersection_size": intersection_size,
            "union_size": union_size,
            "set1_size": len(set1),
            "set2_size": len(set2),
            "minhash_num_perm": self.num_perm
        }
        
        return similarity, details
    
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
        
        # 自定义样式
        liquid.options.get('series')[0].update(
            itemStyle_opts=opts.ItemStyleOpts(
                opacity=0.8,
                shadow_blur=10,
                shadow_color="rgba(0, 0, 0, 0.4)"
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
    
    def forward(
        self, 
        group1_df: pl.LazyFrame, 
        group2_df: pl.LazyFrame
    ) -> Dict:
        """计算两个客户群的相似度"""
        self.logger.info("开始计算客户群体相似度...")
        
        # 并行处理两个群体的特征
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._create_feature_set, group1_df)
            future2 = executor.submit(self._create_feature_set, group2_df)
            
            set1 = future1.result()
            set2 = future2.result()
        
        # 计算相似度
        similarity, details = self._calculate_similarity(set1, set2)
        
        # 创建相似度图表
        chart_options = self._create_similarity_chart(similarity)
        
        # 准备结果
        result = {
            "similarity_score": similarity,
            "details": details,
            "performance_info": {
                "sample_size": self.sample_size,
                "threads_used": self.n_threads,
                "cache_info": self._get_feature_hash.cache_info()._asdict()
            }
        }
        
        # 添加图表到summary
        self.summary = [chart_options]
        
        self.logger.info(f"相似度计算完成: {similarity:.4f}")
        
        return result

