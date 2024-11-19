import polars as pl
import numpy as np
from stages.distance_match import DistanceMatch


def generate_test_data(n_samples_A=1000, n_samples_B=2000):
    """生成测试数据"""
    np.random.seed(42)
    
    # 生成A组数据
    data_A = {
        'user_id': [f'A_{i}' for i in range(n_samples_A)],
        'age': np.random.normal(30, 5, n_samples_A),
        'income': np.random.lognormal(10, 0.5, n_samples_A),
        'purchase_freq': np.random.gamma(2, 2, n_samples_A),
        'psm_score': np.random.beta(2, 2, n_samples_A)
    }
    
    # 生成B组数据
    data_B = {
        'user_id': [f'B_{i}' for i in range(n_samples_B)],
        'age': np.random.normal(32, 6, n_samples_B),
        'income': np.random.lognormal(10.2, 0.6, n_samples_B),
        'purchase_freq': np.random.gamma(2.2, 2.1, n_samples_B),
        'psm_score': np.random.beta(2.2, 2.1, n_samples_B)
    }
    
    return pl.DataFrame(data_A).lazy(), pl.DataFrame(data_B).lazy()


def test_matching_methods():
    """测试不同的匹配方法"""
    print("生成测试数据...")
    lf_A, lf_B = generate_test_data()
    
    feature_cols = ['age', 'income', 'purchase_freq']
    methods = ['nearest', 'radius', 'kernel', 'stratified']
    
    for method in methods:
        print(f"\n测试 {method} 匹配方法...")
        
        matcher = DistanceMatch(
            cols=feature_cols,
            proba_col='psm_score',
            method=method,
            id_col='user_id',
            caliper=0.2,
            k=1 if method == 'nearest' else None,
            need_normalize=True
        )
        
        try:
            result = matcher.forward(lf_A, lf_B).collect()
            
            print(f"匹配结果数量: {len(result)}")
            
            if len(result) > 0:
                # 获取匹配后的数据
                matched_A = (
                    lf_A.filter(
                        pl.col('user_id').is_in(result['id_A_matched'].to_list())
                    ).collect()
                )
                
                matched_B = (
                    lf_B.filter(
                        pl.col('user_id').is_in(result['id_B_matched'].to_list())
                    ).collect()
                )
                
                # 计算匹配前后的特征差异
                print("\n特征均值对比:")
                for col in feature_cols:
                    mean_A_before = lf_A.select(pl.col(col)).collect().mean().item()
                    mean_B_before = lf_B.select(pl.col(col)).collect().mean().item()
                    mean_A_after = matched_A[col].mean()
                    mean_B_after = matched_B[col].mean()
                    
                    print(f"\n{col}:")
                    print(f"  匹配前 - A组: {mean_A_before:.2f}, B组: {mean_B_before:.2f}, "
                          f"差异: {abs(mean_A_before - mean_B_before):.2f}")
                    print(f"  匹配后 - A组: {mean_A_after:.2f}, B组: {mean_B_after:.2f}, "
                          f"差异: {abs(mean_A_after - mean_B_after):.2f}")
            
        except Exception as e:
            print(f"匹配过程出错: {str(e)}")
            raise e
        

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    # 测试空数据集
    empty_df = pl.DataFrame({
        'user_id': [],
        'age': [],
        'income': [],
        'purchase_freq': [],
        'psm_score': []
    }).lazy()
    
    # 测试含有缺失值的数据集
    df_with_nulls = pl.DataFrame({
        'user_id': ['A_1', 'A_2', 'A_3'],
        'age': [25, None, 35],
        'income': [50000, 60000, None],
        'purchase_freq': [10, 20, 30],
        'psm_score': [0.5, 0.6, 0.7]
    }).lazy()
    
    matcher = DistanceMatch(
        cols=['age', 'income', 'purchase_freq'],
        proba_col='psm_score',
        method='nearest',
        id_col='user_id'
    )
    
    # 测试空数据集
    print("\n测试空数据集:")
    try:
        result = matcher.forward(empty_df, empty_df).collect()
        print(f"空数据集匹配结果数量: {len(result)}")
    except Exception as e:
        print(f"空数据集测试出错: {str(e)}")
        raise e
    
    # 测试含有缺失值的数据集
    print("\n测试含有缺失值的数据集:")
    try:
        result = matcher.forward(df_with_nulls, df_with_nulls).collect()
        print(f"含缺失值数据集匹配结果数量: {len(result)}")
    except Exception as e:
        print(f"缺失值测试出错: {str(e)}")
        raise e
    

if __name__ == "__main__":
    print("开始测试距离匹配功能...")
    test_matching_methods()
    test_edge_cases()
    print("\n测试完成!")