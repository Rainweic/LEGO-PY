import pandas as pd


from dags.stage import stage
from dags.pipeline import Pipeline
from stages import *


@stage(n_outputs=1)
def filter_sample_label(df):
    """
    过滤和采样标签数据。
    
    参数:
        df (pd.DataFrame): 输入的数据框。
    
    返回:
        pd.DataFrame: 处理后的数据框。
    """
    # 过滤掉label_value==-1 && 只选取W101
    df = df[df["label_value"] != -1]
    df = df[df["position_id"] == 'W101']
    
    # 正负样本采样1:1
    df_label_1 = df[df["label_value"] == 1]
    df_label_0 = df[df["label_value"] == 0]

    # 正负样本1:1
    n_samples = min(df_label_0.shape[0], df_label_1.shape[0])
    df = pd.concat([df_label_0.sample(n_samples), df_label_1.sample(n_samples)], axis=0)

    df = df[["member_id", "label_value", "dt"]]

    return df


def build_pipeline():
    """
    构建处理流水线。
    
    返回:
        Pipeline: 构建好的流水线对象。
    """

    """--------------label生成-------------"""
    # 读取标签数据
    label_data_path = f"/projects/growth/test/analysis_tools/alg_label/coupon_planned_strategy/expose_label/single/{sample_date}"
    label_data_read_stage = HDFSCSVReadStage(path=label_data_path, overwrite=False)

    # 对标签进行映射
    map = {
        1: 0,   # 弹窗未下单
        2: 0,   # 弹窗三方下单
        0: 1,   # 弹出的券下单
        -1: -1  # 稍后过滤掉
    }
    label_cast_stage = CastStage(feature_name="label_value", map=map) \
                            .after(label_data_read_stage) \
                            .set_input(label_data_read_stage.output_data_names[0])

    
    final_label_stage = filter_sample_label() \
        .after(label_cast_stage) \
        .set_input(label_cast_stage.output_data_names[0])
    

    """-------------获取各类标签-------------"""

    # lucky_member_group_type 客群分类标签
    path = f"/projects/growth/prod/user-reach-label/data/order/order_message_part2/{sample_date}"
    read_data_stage_1 = HDFSORCReadStage(path=path, select_cols=["member_id", "lucky_member_group_type"], overwrite=False)

    # is_alt_member 是否小号
    path = f"/projects/growth/prod/user-reach-label/view/label_view/{sample_date}"
    read_data_stage_2 = HDFSORCReadStage(path=path, select_cols=["member_id", "is_alt_member"], overwrite=False)

    # start_days_30d 用户启动天数_30天
    path = f"/projects/growth/prod/user-reaxch-label/data/coupon-control/base/login_order_labels/{sample_date}"
    read_data_stage_3 = HDFSORCReadStage(path=path, select_cols=["member_id", "start_days_30d"], overwrite=False)

    # is_start_no_order_prefer_user is_coupon_list_bw_no_order_prefer_user
    path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/login_order_labels/{sample_date}"
    read_data_stage_4 = HDFSORCReadStage(path=path, 
                                         select_cols=[
                                             "member_id",
                                             "is_start_no_order_prefer_user",            # 有无启动不下单偏好
                                             "is_coupon_list_bw_no_order_prefer_user"    # 有无浏览优惠券列表不下单偏好
                                         ],
                                         overwrite=False)

    # psm_360d psm_360天
    path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/member_income_psm_360d/{sample_date}"
    read_data_stage_5 =  HDFSORCReadStage(path=path, select_cols=["member_id", "psm_360d"], overwrite=False)

    # app_coupon_list_bw_days_30d 优惠券列表曝光天数_30天
    path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/coupon_bw_labels/{sample_date}"
    read_data_stage_6 = HDFSORCReadStage(path=path, select_cols=["member_id", "app_coupon_list_bw_days_30d"], overwrite=False)

    # coupon_used_99_times all_coupon_used_times
    path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/member_use_99_coupon_180d/{sample_date}"
    read_data_stage_7 = HDFSORCReadStage(path=path,
                                         select_cols=[
                                            "member_id",
                                            "coupon_used_99_times",      # 店庆券使用次数_180天
                                            "all_coupon_used_times"      # 优惠券使用次数_180天
                                         ],
                                         overwrite=False)

    
    # active_user_churn_label 流失概率预测等级
    active_user_churn_label_path = f"/projects/growth/prod/wakeup_model/churn_prob/predict_result/{sample_date}"
    read_data_stage_8 = HDFSCSVReadStage(path=active_user_churn_label_path, select_cols=["active_user_churn_label"], overwrite=False)

    # join到label上
    join_stage = multi_join(on=["member_id"], how="left").set_inputs([
        final_label_stage.output_data_names[0],
        read_data_stage_1.output_data_names[0],
        read_data_stage_2.output_data_names[0],
        read_data_stage_3.output_data_names[0],
        read_data_stage_4.output_data_names[0],
        read_data_stage_5.output_data_names[0],
        read_data_stage_6.output_data_names[0],
        read_data_stage_7.output_data_names[0],
        read_data_stage_8.output_data_names[0]
    ])\
    .after([
        final_label_stage,
        read_data_stage_1,
        read_data_stage_2,
        read_data_stage_3,
        read_data_stage_4,
        read_data_stage_5,
        read_data_stage_6,
        read_data_stage_7,
        read_data_stage_8
    ])
    
    pipeline = Pipeline()
    pipeline.add_stages([
        label_data_read_stage,
        label_cast_stage,
        final_label_stage,
        read_data_stage_1,
        read_data_stage_2,
        read_data_stage_3,
        read_data_stage_4,
        read_data_stage_5,
        read_data_stage_6,
        read_data_stage_7,
        read_data_stage_8,
        join_stage,
    ])

    return pipeline


# 运行
if __name__ == "__main__":
    sample_date = "2024-09-05"
    pipeline = build_pipeline()
    pipeline.start()
