from dags.stage import CustomStage
from typing import Dict
import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasketch import MinHash, MinHashLSH
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


class CustomerSimilarityStage(CustomStage):
    """使用MinHash LSH计算客户群体相似度"""
    
    def __init__(
        self, 
        feature_cols: list[str], 
        num_perm: int = 128,      # MinHash排列数
        threshold: float = 0.01,   # LSH阈值，设置很小以捕获差异
        n_threads: int = 4        # 并行线程数
    ):
        super().__init__(n_outputs=0)
        self.feature_cols = feature_cols
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_threads = n_threads
        
    @lru_cache(maxsize=1024)
    def _encode_feature(self, feature_value: str) -> str:
        """对单个特征值进行编码，使用缓存加速"""
        return f"{hash(feature_value):x}"
        
    def _create_minhash(self, features: list[str]) -> MinHash:
        """创建MinHash"""
        mh = MinHash(num_perm=self.num_perm)
        for f in features:
            mh.update(self._encode_feature(f).encode('utf8'))
        return mh
        
    def _process_batch(self, records: list[dict], feature_cols: list[str]) -> list[MinHash]:
        """并行处理一批记录"""
        minhashes = []
        for record in records:
            features = [str(record.get(col, 'MISSING')) for col in feature_cols]
            mh = self._create_minhash(features)
            minhashes.append(mh)
        return minhashes
        
    def forward(
        self, 
        group1_df: pl.LazyFrame, 
        group2_df: pl.LazyFrame
    ) -> Dict:
        """计算两个客户群的相似度"""
        self.logger.info("开始计算客户群体相似度...")
        
        # 收集数据并转换为字典列表
        df1 = group1_df.collect()
        df2 = group2_df.collect()
        
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
                    "cache_info": self._encode_feature.cache_info()._asdict()
                }
            }
        
        records1 = df1.to_dicts()
        records2 = df2.to_dicts()
        
        # 分批处理
        batch_size = max(100, min(len(records1), len(records2)) // self.n_threads)
        batches1 = [records1[i:i + batch_size] for i in range(0, len(records1), batch_size)]
        batches2 = [records2[i:i + batch_size] for i in range(0, len(records2), batch_size)]
        
        # 并行创建MinHash
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            minhashes1 = []
            minhashes2 = []
            
            # 并行处理两组数据
            futures1 = [
                executor.submit(self._process_batch, batch, self.feature_cols)
                for batch in batches1
            ]
            futures2 = [
                executor.submit(self._process_batch, batch, self.feature_cols)
                for batch in batches2
            ]
            
            # 收集结果
            for future in futures1:
                minhashes1.extend(future.result())
            for future in futures2:
                minhashes2.extend(future.result())
        
        # 使用LSH计算相似度
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        
        # 添加第一组的MinHash
        for i, mh in enumerate(minhashes1):
            lsh.insert(f"g1_{i}", mh)
        
        # 查询第二组并计算相似度
        similarities = []
        for mh2 in minhashes2:
            result = lsh.query(mh2)
            if result:
                # 计算与匹配项的最大相似度
                max_sim = max(
                    mh2.jaccard(minhashes1[int(r.split('_')[1])])
                    for r in result
                )
                similarities.append(max_sim)
            else:
                similarities.append(0.0)
        
        final_similarity = float(np.mean(similarities)) if similarities else 0.0
        
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
                "cache_info": self._encode_feature.cache_info()._asdict()
            }
        }
        
        self.logger.info(f"相似度计算完成: {result}")
        return result
