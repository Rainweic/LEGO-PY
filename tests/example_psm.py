import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

def psm_workflow(data_A, data_B, features):
    """
    PSM工作流程
    
    Args:
        data_A: 处理组数据（例如：参与营销活动的用户）
        data_B: 对照组数据（例如：未参与营销活动的用户）
        features: 用于匹配的特征列表
    """
    # 1. 数据预处理
    scaler = StandardScaler()
    X = pd.concat([data_A[features], data_B[features]])
    X_scaled = scaler.fit_transform(X)
    
    # 构建标签：处理组为1，对照组为0
    y = np.concatenate([
        np.ones(len(data_A)), 
        np.zeros(len(data_B))
    ])
    
    # 2. 训练倾向性模型
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # 计算倾向性得分
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    
    # 3. 评估群体相似度
    auc = roc_auc_score(y, propensity_scores)
    print(f"群体相似度评估 (AUC): {auc:.4f}")
    print(f"AUC接近0.5表示两组更相似，接近1表示差异较大")
    
    # 4. 根据倾向性得分进行匹配
    matched_pairs = match_groups(
        data_A, data_B, 
        propensity_scores[:len(data_A)],
        propensity_scores[len(data_A):],
        caliper=0.2
    )
    
    return matched_pairs, propensity_scores

def match_groups(data_A, data_B, scores_A, scores_B, caliper=0.2):
    """
    使用最近邻匹配方法进行组间匹配
    
    Args:
        caliper: 匹配阈值，超过此阈值的样本对将被排除
    """
    # 计算处理组和对照组之间的距离矩阵
    distances = cdist(
        scores_A.reshape(-1, 1), 
        scores_B.reshape(-1, 1)
    )
    
    matched_pairs = []
    used_B = set()
    
    # 为每个处理组样本找到最近的对照组样本
    for i in range(len(data_A)):
        min_dist = float('inf')
        best_match = None
        
        for j in range(len(data_B)):
            if j not in used_B and distances[i, j] < min_dist:
                if distances[i, j] <= caliper:  # 只匹配在阈值范围内的样本
                    min_dist = distances[i, j]
                    best_match = j
        
        if best_match is not None:
            matched_pairs.append((i, best_match))
            used_B.add(best_match)
    
    return matched_pairs

# 示例：评估营销活动效果
def marketing_campaign_example():
    """营销活动效果评估示例"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 处理组：参与营销活动的用户
    campaign_users = pd.DataFrame({
        'age': np.random.normal(35, 8, n_samples),  # 年龄
        'consumption': np.random.lognormal(4, 0.5, n_samples),  # 历史消费
        'activity': np.random.gamma(2, 2, n_samples),  # 活跃度
        'conversion': np.random.binomial(1, 0.3, n_samples)  # 转化情况
    })
    
    # 对照组：未参与营销活动的用户
    control_users = pd.DataFrame({
        'age': np.random.normal(30, 10, n_samples * 2),  # 略有差异的年龄分布
        'consumption': np.random.lognormal(3.8, 0.6, n_samples * 2),  # 略低的消费水平
        'activity': np.random.gamma(1.8, 2, n_samples * 2),  # 略低的活跃度
        'conversion': np.random.binomial(1, 0.2, n_samples * 2)  # 转化率
    })
    
    # 执行PSM
    features = ['age', 'consumption', 'activity']
    matched_pairs, scores = psm_workflow(campaign_users, control_users, features)
    
    # 分析匹配效果
    print(f"\n成功匹配的样本对数量: {len(matched_pairs)}")
    
    # 计算活动效果
    campaign_conversion = campaign_users.iloc[[i for i, _ in matched_pairs]]['conversion'].mean()
    control_conversion = control_users.iloc[[j for _, j in matched_pairs]]['conversion'].mean()
    
    print(f"\n营销活动效果分析:")
    print(f"处理组转化率: {campaign_conversion:.4f}")
    print(f"对照组转化率: {control_conversion:.4f}")
    print(f"提升效果: {(campaign_conversion - control_conversion) / control_conversion * 100:.2f}%")

if __name__ == "__main__":
    marketing_campaign_example()