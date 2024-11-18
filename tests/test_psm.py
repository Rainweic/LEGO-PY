import polars as pl
import numpy as np
from stages.psm import PSM

def generate_test_data(n_samples_A=1000, n_samples_B=2000, n_features=5, similarity=0.8):
    """
    生成测试数据
    
    Args:
        n_samples_A: 群体A的样本量
        n_samples_B: 群体B的样本量
        n_features: 特征数量
        similarity: 两组数据的相似度(0-1之间，越大表示越相似)
    """
    # 为群体A生成数据
    mean_A = np.random.randn(n_features)
    cov_A = np.eye(n_features)  # 使用单位矩阵作为协方差矩阵
    data_A = np.random.multivariate_normal(mean_A, cov_A, n_samples_A)
    
    # 为群体B生成相似但不完全相同的数据
    mean_B = mean_A * similarity + np.random.randn(n_features) * (1 - similarity)
    cov_B = cov_A * similarity + np.eye(n_features) * (1 - similarity)
    data_B = np.random.multivariate_normal(mean_B, cov_B, n_samples_B)
    
    # 转换为Polars DataFrame
    cols = [f"feature_{i+1}" for i in range(n_features)]
    
    df_A = pl.DataFrame(
        data_A,
        schema=cols
    )
    
    df_B = pl.DataFrame(
        data_B,
        schema=cols
    )
    
    return df_A.lazy(), df_B.lazy()

def test_psm():
    """测试PSM类的功能"""
    
    # 1. 生成测试数据
    print("正在生成测试数据...")
    lz_A, lz_B = generate_test_data(
        n_samples_A=1000,    # 群体A样本量
        n_samples_B=2000,    # 群体B样本量
        n_features=5,        # 特征数量
        similarity=0.8       # 设置的相似度
    )
    
    # 2. 创建PSM实例
    feature_cols = [f"feature_{i+1}" for i in range(5)]
    psm = PSM(
        cols=feature_cols,
        need_normalize=True,
        similarity_method='cosine',
        model_params={
            'random_state': 42,
            'max_iter': 1000
        }
    )
    
    # 3. 运行PSM评估
    print("开始PSM评估...")
    try:
        result = psm.forward(lz_A, lz_B)
        print("PSM评估完成！")
        print("可视化结果已保存为 psm_distribution.html 和 psm_metrics.html")
        
        # 4. 打印一些基本统计信息
        print("\n数据统计信息：")
        print(f"群体A样本量: {lz_A.select(pl.count()).collect().item()}")
        print(f"群体B样本量: {lz_B.select(pl.count()).collect().item()}")
        
        # 5. 检查特征分布
        print("\n特征分布概览：")
        stats_A = lz_A.select([
            pl.col(col).mean().alias(f"{col}_mean_A")
            for col in feature_cols
        ]).collect()
        
        stats_B = lz_B.select([
            pl.col(col).mean().alias(f"{col}_mean_B")
            for col in feature_cols
        ]).collect()
        
        for col in feature_cols:
            mean_A = stats_A[f"{col}_mean_A"][0]
            mean_B = stats_B[f"{col}_mean_B"][0]
            print(f"{col}: 群体A均值={mean_A:.4f}, 群体B均值={mean_B:.4f}")
        
    except Exception as e:
        print(f"PSM评估过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_psm()