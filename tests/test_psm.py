import polars as pl
import numpy as np
from stages.psm import PSM

def generate_test_data(n_samples_A=1000, n_samples_B=2000, similarity=0.8):
    """生成更贴近真实场景的测试数据
    
    特征包含：
    - 连续特征：年龄、消费金额、访问频次等
    - 比率特征：转化率、点击率等
    - 计数特征：订单数、访问次数等
    - 时间特征：首次访问时间、最近访问间隔等
    """
    np.random.seed(42)
    
    # 1. 连续特征生成
    age_A = np.random.normal(30, 8, n_samples_A)  # 年龄分布
    age_B = np.random.normal(30 * similarity + 5 * (1-similarity), 8, n_samples_B)
    
    amount_A = np.random.lognormal(4, 1, n_samples_A)  # 消费金额
    amount_B = np.random.lognormal(4 * similarity + 0.5 * (1-similarity), 1, n_samples_B)
    
    frequency_A = np.random.gamma(2, 2, n_samples_A)  # 访问频次
    frequency_B = np.random.gamma(2 * similarity + 1 * (1-similarity), 2, n_samples_B)
    
    # 2. 比率特征生成
    cvr_A = np.random.beta(2, 5, n_samples_A)  # 转化率
    cvr_B = np.random.beta(2 * similarity + 1 * (1-similarity), 5, n_samples_B)
    
    ctr_A = np.random.beta(1, 10, n_samples_A)  # 点击率
    ctr_B = np.random.beta(1 * similarity + 0.5 * (1-similarity), 10, n_samples_B)
    
    # 3. 计数特征生成
    orders_A = np.random.poisson(5, n_samples_A)  # 订单数
    orders_B = np.random.poisson(5 * similarity + 2 * (1-similarity), n_samples_B)
    
    visits_A = np.random.poisson(10, n_samples_A)  # 访问次数
    visits_B = np.random.poisson(10 * similarity + 3 * (1-similarity), n_samples_B)
    
    # 4. 时间特征生成（以天为单位）
    first_visit_A = np.random.uniform(0, 365, n_samples_A)  # 首次访问距今天数
    first_visit_B = np.random.uniform(0 * similarity + 30 * (1-similarity), 365, n_samples_B)
    
    last_interval_A = np.random.exponential(30, n_samples_A)  # 最近访问间隔
    last_interval_B = np.random.exponential(30 * similarity + 10 * (1-similarity), n_samples_B)
    
    # 构建DataFrame
    df_A = pl.DataFrame({
        'age': age_A,
        'consumption_amount': amount_A,
        'visit_frequency': frequency_A,
        'conversion_rate': cvr_A,
        'click_rate': ctr_A,
        'order_count': orders_A,
        'visit_count': visits_A,
        'first_visit_days': first_visit_A,
        'last_visit_interval': last_interval_A
    })
    
    df_B = pl.DataFrame({
        'age': age_B,
        'consumption_amount': amount_B,
        'visit_frequency': frequency_B,
        'conversion_rate': cvr_B,
        'click_rate': ctr_B,
        'order_count': orders_B,
        'visit_count': visits_B,
        'first_visit_days': first_visit_B,
        'last_visit_interval': last_interval_B
    })
    
    return df_A.lazy(), df_B.lazy()

def test_psm():
    """测试PSM类的功能"""
    print("正在生成测试数据...")
    
    # 定义特征列
    feature_cols = [
        'age', 'consumption_amount', 'visit_frequency',
        'conversion_rate', 'click_rate', 'order_count',
        'visit_count', 'first_visit_days', 'last_visit_interval'
    ]
    
    lz_A, lz_B = generate_test_data(
        n_samples_A=1000,
        n_samples_B=2000,
        similarity=0.8
    )
    
    psm = PSM(
        cols=feature_cols,
        need_normalize=True,
        similarity_method='cosine'
    )
    
    print("开始PSM评估...")
    try:
        result = psm.forward(lz_A, lz_B)
        
        # 打印详细的统计信息
        print("\n特征分布概览：")
        
        # 构建统计表达式列表
        stats_exprs_A = []
        stats_exprs_B = []
        for col in feature_cols:
            stats_exprs_A.extend([
                pl.col(col).mean().alias(f"{col}_mean_A"),
                pl.col(col).std().alias(f"{col}_std_A"),
                pl.col(col).median().alias(f"{col}_median_A")
            ])
            stats_exprs_B.extend([
                pl.col(col).mean().alias(f"{col}_mean_B"),
                pl.col(col).std().alias(f"{col}_std_B"),
                pl.col(col).median().alias(f"{col}_median_B")
            ])
        
        # 计算统计量
        stats_A = lz_A.select(stats_exprs_A).collect()
        stats_B = lz_B.select(stats_exprs_B).collect()
        
        # 打印每个特征的统计信息
        for col in feature_cols:
            print(f"\n{col}统计信息:")
            print(f"群体A - 均值: {stats_A[f'{col}_mean_A'][0]:.4f}, "
                  f"标准差: {stats_A[f'{col}_std_A'][0]:.4f}, "
                  f"中位数: {stats_A[f'{col}_median_A'][0]:.4f}")
            print(f"群体B - 均值: {stats_B[f'{col}_mean_B'][0]:.4f}, "
                  f"标准差: {stats_B[f'{col}_std_B'][0]:.4f}, "
                  f"中位数: {stats_B[f'{col}_median_B'][0]:.4f}")
        
    except Exception as e:
        print(f"PSM评估过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_psm()