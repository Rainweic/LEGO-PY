import polars as pl
import numpy as np
from scipy import stats
from stages.psm import PSM
from stages.distance_match import DistanceMatch

def generate_test_data(n_samples_A=1000, n_samples_B=2000, similarity=0.8):
    """生成测试数据
    
    Args:
        similarity: 相似度参数(0-1), 值越大表示两组数据越相似
            - 0.9: 极其相似，特征分布几乎重叠
            - 0.6: 中等相似，特征分布有一定差异
            - 0.3: 差异较大，特征分布明显不同
    """
    np.random.seed(42)
    
    # 1. 连续特征生成
    age_params = [30, 8]  # 均值, 标准差
    amount_params = [4, 1]  # log均值, log标准差
    freq_params = [2, 2]  # shape, scale
    
    # 生成A组特征
    age_A = np.random.normal(*age_params, n_samples_A)
    amount_A = np.random.lognormal(*amount_params, n_samples_A)
    frequency_A = np.random.gamma(*freq_params, n_samples_A)
    
    # 生成B组特征(根据相似度调整参数)
    age_params_B = [p * (1 + (1-similarity) * 0.2) for p in age_params]
    amount_params_B = [p * (1 + (1-similarity) * 0.1) for p in amount_params]
    freq_params_B = [p * (1 + (1-similarity) * 0.15) for p in freq_params]
    
    age_B = np.random.normal(*age_params_B, n_samples_B)
    amount_B = np.random.lognormal(*amount_params_B, n_samples_B)
    frequency_B = np.random.gamma(*freq_params_B, n_samples_B)
    
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
        'id': np.arange(n_samples_A),  # 添加id列
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
        'id': np.arange(n_samples_B),  # 添加id列
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
    feature_cols = [
        'age', 'consumption_amount', 'visit_frequency',
        'conversion_rate', 'click_rate', 'order_count',
        'visit_count', 'first_visit_days', 'last_visit_interval'
    ]
    
    # 生成不同相似度的测试数据
    test_scenarios = [
        (0.9, "高相似度"),  # 期望PSM效果好
        (0.6, "中等相似度"), # 期望PSM效果一般
        (0.3, "低相似度")   # 期望PSM效果差
    ]
    
    test_data = [
        generate_test_data(
            n_samples_A=1000,
            n_samples_B=2000,
            similarity=sim
        ) for sim, _ in test_scenarios
    ]
    
    # 测试不同的匹配方法
    match_methods = ['nearest', 'radius', 'kernel', 'stratified']
    
    for (lz_A, lz_B), (sim, desc) in zip(test_data, test_scenarios):
        # 1. 先测试PSM模型
        psm = PSM(
            cols=feature_cols,
            need_normalize=True,
            similarity_method='cosine'
        )
        
        # 执行PSM得分计算
        lz_A_scored, lz_B_scored = psm.forward(lz_A, lz_B)
        
        # 验证PSM模型效果
        if sim >= 0.9:
            # 高相似度数据的期望：AUC接近0.5，KS很小，重叠度很高
            assert psm.metrics['auc'] < 0.65, f"高相似度数据的AUC过高: {psm.metrics['auc']:.4f}"
            assert psm.metrics['ks'] < 0.15, f"高相似度数据的KS过高: {psm.metrics['ks']:.4f}"
            assert psm.metrics['overlap'] > 0.7, f"高相似度数据的重叠度过低: {psm.metrics['overlap']:.4f}"
        elif sim >= 0.6:
            # 中等相似度数据的期望
            assert psm.metrics['auc'] < 0.75, f"中等相似度数据的AUC过高: {psm.metrics['auc']:.4f}"
            assert psm.metrics['ks'] < 0.25, f"中等相似度数据的KS过高: {psm.metrics['ks']:.4f}"
            assert psm.metrics['overlap'] > 0.5, f"中等相似度数据的重叠度过低: {psm.metrics['overlap']:.4f}"
        else:
            # 低相似度数据的期望
            assert psm.metrics['auc'] < 0.85, f"低相似度数据的AUC过高: {psm.metrics['auc']:.4f}"
            assert psm.metrics['ks'] < 0.35, f"低相似度数据的KS过高: {psm.metrics['ks']:.4f}"
            assert psm.metrics['overlap'] > 0.3, f"低相似度数据的重叠度过低: {psm.metrics['overlap']:.4f}"
        
        # 2. 测试不同的匹配方法
        for method in match_methods:
            # 根据特征重要性选择匹配特征
            important_features = [
                col for col, importance in psm.feature_importance.items() 
                if importance > 0.05  # 选择重要性大于5%的特征
            ]
            
            matcher = DistanceMatch(
                cols=important_features,  # 使用重要特征进行匹配
                proba_col="psm_score",  # 同时使用PS得分
                id_col="id",
                method=method,
                caliper=0.25,
                k=1 if method != 'radius' else 3,
                need_normalize=True  # 因为使用了原始特征，需要标准化
            )
            
            # 执行匹配
            matched_result = matcher.forward(lz_A_scored, lz_B_scored)
            matched_df = matched_result.collect()
            
            # 验证匹配结果
            assert len(matched_df) > 0, f"{method}方法在{desc}场景下未能匹配到任何样本"
            
            # 计算匹配率
            match_rate_A = len(matched_df) / lz_A.select(pl.count()).collect().item()
            match_rate_B = len(matched_df) / lz_B.select(pl.count()).collect().item()
            
            # 根据相似度验证匹配率
            if sim >= 0.9:
                assert match_rate_A >= 0.7, f"{method}方法在高相似度下匹配率过低: {match_rate_A:.2%}"
            elif sim >= 0.6:
                assert match_rate_A >= 0.5, f"{method}方法在中等相似度下匹配率过低: {match_rate_A:.2%}"
            else:
                assert match_rate_A >= 0.3, f"{method}方法在低相似度下匹配率过低: {match_rate_A:.2%}"
            
            # 验证匹配后的样本平衡性
            matched_A = lz_A_scored.filter(
                pl.col("id").is_in(matched_df["id_A_matched"])
            ).collect()
            matched_B = lz_B_scored.filter(
                pl.col("id").is_in(matched_df["id_B_matched"])
            ).collect()
            
            # 验证特征平衡性
            for col in feature_cols:
                ks_stat, _ = stats.ks_2samp(
                    matched_A[col].to_numpy(),
                    matched_B[col].to_numpy()
                )
                
                # 根据相似度验证KS统计量
                if sim >= 0.9:
                    assert ks_stat < 0.1, f"{method}方法在高相似度下{col}特征的KS统计量过高: {ks_stat:.4f}"
                elif sim >= 0.6:
                    assert ks_stat < 0.2, f"{method}方法在中等相似度下{col}特征的KS统计量过高: {ks_stat:.4f}"
                else:
                    assert ks_stat < 0.3, f"{method}方法在低相似度下{col}特征的KS统计量过高: {ks_stat:.4f}"
            
            # 验证匹配ID的唯一性(1:1匹配)
            if method != 'radius':
                assert len(matched_df["id_A_matched"].unique()) == len(matched_df), (
                    f"{method}方法产生了重复的处理组匹配"
                )
                assert len(matched_df["id_B_matched"].unique()) == len(matched_df), (
                    f"{method}方法产生了重复的对照组匹配"
                )

if __name__ == "__main__":
    test_psm()