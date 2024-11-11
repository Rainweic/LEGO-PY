from dags.stage import CustomStage
from typing import Dict
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasketch import MinHash, MinHashLSH
from multiprocessing import Pool
from functools import lru_cache
from pyecharts import options as opts
from pyecharts.charts import Liquid



class CustomerSimilarityStage(CustomStage):
    """使用MinHash LSH计算客户群体相似度"""
    
    def __init__(
        self, 
        feature_cols: list[str], 
        weights: dict[str, int] | str = "auto",     # 可以是具体的权重dict或"auto"
        default_weight: int = 1,                    # 若weights中未覆盖所有权重，则未设置的特征所对应的权重默认值
        num_perm: int = 128,                        # MinHash排列数
        threshold: float = 0.01,                    # LSH阈值，设置很小以捕获差异
        n_threads: int = 6                          # 并行线程数
    ):
        super().__init__(n_outputs=0)
        self.feature_cols = feature_cols
        self.weights_config = weights
        self.default_weight = default_weight
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_threads = n_threads
        
    @lru_cache(maxsize=1024)
    def _encode_feature(self, feature_value: str) -> str:
        """对单个特征值进行编码，使用缓存加速"""
        return f"{hash(feature_value):x}"
        
    @staticmethod
    def _process_batch_wrapper(args):
        """包装_process_batch方法以支持单参数"""
        records, feature_cols, weights = args
        return CustomerSimilarityStage._process_batch(records, feature_cols, weights)
        
    @staticmethod
    def _process_batch(records: list[dict], feature_cols: list[str], weights: dict) -> list[MinHash]:
        """并行处理一批记录，加入权重参数"""
        minhashes = []
        for record in records:
            features = [str(record.get(col, 'MISSING')) for col in feature_cols]
            mh = CustomerSimilarityStage._create_minhash(features, weights, feature_cols)
            minhashes.append(mh)
        return minhashes
        
    @staticmethod
    def _create_minhash(features: list[str], weights: dict, feature_cols: list[str]) -> MinHash:
        """创建MinHash，使用权重"""
        mh = MinHash(num_perm=128)  # 使用固定值或通过参数传递
        
        # 对每个特征单独处理
        for col, val in zip(feature_cols, features):
            weight = weights.get(col, 1)  # 使用默认权重1
            feature_str = f"{col}:{val}".encode('utf8')
            for _ in range(int(weight)):
                mh.update(feature_str)
        
        return mh
        
    def _validate_weights(self, weights: dict) -> None:
        """验证权重配置"""
        # 检查是否有未知特征
        unknown_features = set(weights.keys()) - set(self.feature_cols)
        if unknown_features:
            self.logger.warning(f"权重配置中包含未知特征: {unknown_features}")
            
        # 检查是否有特征未配置权重
        missing_features = set(self.feature_cols) - set(weights.keys())
        if missing_features:
            self.logger.warning(f"以下特征未配置权重，将使用默认值{self.default_weight}: {missing_features}")
            
        # 检查权重值的合法性
        invalid_weights = {k: v for k, v in weights.items() if not isinstance(v, (int, float)) or v <= 0}
        if invalid_weights:
            self.logger.warning(f"发现无效的权重值: {invalid_weights}，这些特征将使用默认值{self.default_weight}")
            
    def _get_weights(self, df1: pl.DataFrame, df2: pl.DataFrame) -> dict:
        """获取特征权重"""
        if isinstance(self.weights_config, dict):
            # 验证手动配置的权重
            self._validate_weights(self.weights_config)
            
            # 使用手动配置的权重，对未配置的特征使用默认值
            weights = {
                col: self.weights_config.get(col, self.default_weight) 
                if isinstance(self.weights_config.get(col), (int, float)) and self.weights_config.get(col) > 0
                else self.default_weight
                for col in self.feature_cols
            }
            
            self.logger.info(f"最终使用的特征权重: {weights}")
            return weights
            
        elif self.weights_config == "auto":
            # 自动计算权重
            weights = {}
            for col in self.feature_cols:
                # 计算每个值的频率分布
                dist1 = (df1.select(pl.col(col))
                        .group_by(col)
                        .agg(pl.len())
                        .with_columns(pl.col('len') / df1.height)
                        .sort('len', descending=True))
                
                dist2 = (df2.select(pl.col(col))
                        .group_by(col)
                        .agg(pl.len())
                        .with_columns(pl.col('len') / df2.height)
                        .sort('len', descending=True))
                
                # 获取所有唯一值
                all_values = pl.concat([
                    df1.select(col), 
                    df2.select(col)
                ]).unique().to_series().to_list()
                
                # 创建值到频率的映射
                freq1 = {row[col]: row['len'] for row in dist1.to_dicts()}
                freq2 = {row[col]: row['len'] for row in dist2.to_dicts()}
                
                # 计算分布差异
                total_diff = 0
                for val in all_values:
                    p1 = freq1.get(val, 0.0001)
                    p2 = freq2.get(val, 0.0001)
                    total_diff += abs(p1 - p2)
                    
                # 将差异转换为权重(1-5的范围)
                weight = 1 + int(4 * total_diff)
                weights[col] = min(5, weight)
                
                self.logger.info(f"特征 {col} 的分布差异: {total_diff:.4f}, 权重: {weights[col]}")
                
            self.logger.info(f"自动计算的特征权重: {weights}")
            return weights
        else:
            # 默认所有特征权重相等
            weights = {col: self.default_weight for col in self.feature_cols}
            self.logger.info(f"使用默认特征权重: {weights}")
            return weights
        
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

        
    def forward(self, group1_df: pl.LazyFrame, group2_df: pl.LazyFrame) -> Dict:
        """计算两个客户群的相似度"""
        self.logger.info("开始计算客户群体相似度...")
        
        # 收集数据并转换为字典列表
        df1 = group1_df.collect()
        df2 = group2_df.collect()
        self.logger.info(f"数据集大小 - 群体1: {df1.height}, 群体2: {df2.height}")

        # 获取权重
        weights = self._get_weights(df1, df2)
        
        if df1.height == 0 or df2.height == 0:
            return {
                "similarity_score": 0.0,
                "details": {
                    "intersection_size": 0,
                    "union_size": max(df1.height, df2.height),
                    "vectors1_size": df1.height,
                    "vectors2_size": df2.height
                },
                "performance_info": {
                    "empty_data": True,
                    "cache_info": self._encode_feature.cache_info()._asdict(),
                    "weights": weights
                }
            }
        
        records1 = df1.to_dicts()
        records2 = df2.to_dicts()
        
        # 分批处理
        batch_size = max(100, min(len(records1), len(records2)) // self.n_threads)
        batches1 = [records1[i:i + batch_size] for i in range(0, len(records1), batch_size)]
        batches2 = [records2[i:i + batch_size] for i in range(0, len(records2), batch_size)]
        
        self.logger.info(f"开始并行处理 - 批次大小: {batch_size}, 群体1批次数: {len(batches1)}, 群体2批次数: {len(batches2)}")
        
        # 并行创建MinHash，使用进程池
        with Pool(processes=self.n_threads) as pool:
            # 准备参数
            batch_params1 = [(batch, self.feature_cols, weights) for batch in batches1]
            batch_params2 = [(batch, self.feature_cols, weights) for batch in batches2]
            
            # 并行处理两组数据
            minhashes1 = []
            minhashes2 = []
            
            # 处理第一组数据
            self.logger.info("处理群体1...")
            results1 = []
            for i, result in enumerate(pool.imap(CustomerSimilarityStage._process_batch_wrapper, batch_params1)):
                results1.append(result)
                if (i + 1) % max(1, len(batches1) // 10) == 0:  # 每完成10%输出一次
                    self.logger.info(f"群体1进度: {(i + 1) / len(batches1):.1%}")
            
            # 处理第二组数据
            self.logger.info("处理群体2...")
            results2 = []
            for i, result in enumerate(pool.imap(CustomerSimilarityStage._process_batch_wrapper, batch_params2)):
                results2.append(result)
                if (i + 1) % max(1, len(batches2) // 10) == 0:  # 每完成10%输出一次
                    self.logger.info(f"群体2进度: {(i + 1) / len(batches2):.1%}")
            
            # 收集结果
            for result in results1:
                minhashes1.extend(result)
            for result in results2:
                minhashes2.extend(result)
        
        self.logger.info("开始计算相似度...")
        
        # 使用LSH计算相似度
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        
        # 添加第一组的MinHash
        for i, mh in enumerate(minhashes1):
            lsh.insert(f"g1_{i}", mh)
            if (i + 1) % max(1, len(minhashes1) // 10) == 0:  # 每完成10%输出一次
                self.logger.info(f"LSH索引构建进度: {(i + 1) / len(minhashes1):.1%}")
        
        # 查询第二组并计算相似度
        similarities = []
        for i, mh2 in enumerate(minhashes2):
            result = lsh.query(mh2)
            if result:
                sims = [mh2.jaccard(minhashes1[int(r.split('_')[1])]) for r in result]
                similarities.append(np.mean(sims))
            else:
                similarities.append(0.0)
                
            if (i + 1) % max(1, len(minhashes2) // 10) == 0:  # 每完成10%输出一次
                self.logger.info(f"相似度计算进度: {(i + 1) / len(minhashes2):.1%}")
        
        # 使用非零相似度的平均值
        non_zero_sims = [s for s in similarities if s > 0]
        final_similarity = float(np.mean(non_zero_sims)) if non_zero_sims else 0.0
        
        result = {
            "similarity_score": final_similarity,
            "details": {
                "intersection_size": sum(1 for s in similarities if s > 0),
                "union_size": len(records1) + len(records2),
                "vectors1_size": len(records1),
                "vectors2_size": len(records2)
            },
            "performance_info": {
                "empty_data": False,
                "batch_size": batch_size,
                "num_batches": len(batches1) + len(batches2),
                "cache_info": self._encode_feature.cache_info()._asdict(),
                "weights": weights
            }
        }
        
        self.logger.info(f"相似度计算完成: {result}")
        self.summary = self._create_similarity_chart(final_similarity)
