import dowhy.datasets
import polars as pl
from scipy import stats
from stages.psm import PSM
from stages.distance_match import DistanceMatch


def prepare_lalonde_data():
    """准备LaLonde数据集
    
    Returns:
        tuple: (处理组LazyFrame, 对照组LazyFrame)
    """
    # 加载原始数据
    df = dowhy.datasets.lalonde_dataset()
    
    # 转换为polars DataFrame并添加id列
    df = pl.from_pandas(df)
    df = df.with_columns(pl.Series(range(len(df))).alias("id"))
    
    # 分离处理组和对照组
    nsw_df = df.filter(pl.col("treat") == 1)
    psid_df = df.filter(pl.col("treat") == 0)
    
    # 选择用于匹配的特征列
    feature_cols = [
        'age', 'educ', 'black', 'hisp', 
        'married', 'nodegr', 're74', 're75'
    ]
    
    # 转换为LazyFrame
    nsw_lf = nsw_df.select(["id"] + feature_cols).lazy()
    psid_lf = psid_df.select(["id"] + feature_cols).lazy()
    
    return nsw_lf, psid_lf, feature_cols

def test_psm():
    """测试PSM类的功能"""
    # 准备数据
    lz_A, lz_B, feature_cols = prepare_lalonde_data()
    
    # 1. 测试PSM模型
    psm = PSM(
        cols=feature_cols,
        need_normalize=True,
        similarity_method='cosine'
    )
    
    # 执行PSM得分计算
    lz_A_scored, lz_B_scored = psm.forward(lz_A, lz_B)
    
    # 验证PSM模型效果
    # LaLonde数据集中处理组和对照组差异较大,期望:
    assert 0.65 < psm.metrics['auc'] < 0.85, f"AUC不在合理范围: {psm.metrics['auc']:.4f}"
    assert 0.2 < psm.metrics['ks'] < 0.4, f"KS不在合理范围: {psm.metrics['ks']:.4f}"
    assert psm.metrics['overlap'] > 0.3, f"重叠度过低: {psm.metrics['overlap']:.4f}"
    
    # 2. 测试不同的匹配方法
    match_methods = ['nearest', 'radius', 'kernel', 'stratified']
    
    for method in match_methods:
        # 根据特征重要性选择匹配特征
        important_features = [
            col for col, importance in psm.feature_importance.items() 
            if importance > 0.05
        ]
        
        matcher = DistanceMatch(
            cols=important_features,
            proba_col="psm_score",
            id_col="id",
            method=method,
            caliper=0.25,
            k=1 if method != 'radius' else 3,
            need_normalize=True
        )
        
        # 执行匹配
        matched_result = matcher.forward(lz_A_scored, lz_B_scored)
        matched_df = matched_result.collect()
        
        # 验证匹配结果
        assert len(matched_df) > 0, f"{method}方法未能匹配到样本"
        
        # 计算匹配率(处理组匹配率应该较高)
        match_rate_A = len(matched_df) / lz_A.select(pl.count()).collect().item()
        assert match_rate_A >= 0.5, f"{method}方法处理组匹配率过低: {match_rate_A:.2%}"
        
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
            assert ks_stat < 0.2, f"{method}方法匹配后{col}特征的KS统计量过高: {ks_stat:.4f}"
        
        # 验证匹配ID的唯一性
        if method != 'radius':
            assert len(matched_df["id_A_matched"].unique()) == len(matched_df)
            assert len(matched_df["id_B_matched"].unique()) == len(matched_df)

if __name__ == "__main__":
    test_psm()