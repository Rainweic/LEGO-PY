import logging
import polars as pl
from dags.stage import stage
from dags.pipeline import Pipeline
from stages import *


@stage(n_outputs=1)
def filter_sample_label(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    过滤和采样标签数据。

    参数:
        df (pl.LazyFrame): 输入的数据框。

    返回:
        pl.LazyFrame: 处理后的数据框。
    """

    # 将 DataFrame 转换为 LazyFrame
    df = df.lazy()

    # 过滤掉 label_value == -1 && 只选取 position_id == "W101"
    df = df.filter((pl.col("label_value") != -1) & (pl.col("position_id") == "W101"))

    # 选取所需列
    df = df.select(["member_id", "label_value", "dt"])

    # 正负样本采样1:1
    df_label_1 = df.filter(pl.col("label_value") == 1)
    df_label_0 = df.filter(pl.col("label_value") == 0)

    # 正负样本1:1
    n_samples = min(
        df_label_0.select(pl.len()).collect().item(),
        df_label_1.select(pl.len()).collect().item(),
        n_all_samples // 2
    )

    # 使用 SQL 进行采样(暂时没有随机)
    df_label_0_sampled = df_label_0.sql(f"SELECT * FROM self LIMIT {n_samples}")
    df_label_1_sampled = df_label_1.sql(f"SELECT * FROM self LIMIT {n_samples}")

    # 合并正负样本
    df = pl.concat([df_label_0_sampled, df_label_1_sampled])

    return df.lazy()


if __name__ == "__main__":
    n_all_samples = 200_000
    sample_date = "2024-09-05"

    with Pipeline() as p:
        """
        构建处理流水线。

        返回:
            Pipeline: 构建好的流水线对象。
        """

        """--------------label生成-------------"""
        # 读取标签数据
        label_data_path = f"/projects/growth/test/analysis_tools/alg_label/coupon_planned_strategy/expose_label/single/{sample_date}"
        label_data_read_stage = HDFSCSVReadStage(
            path=label_data_path, overwrite=False
        ).set_pipeline(p)

        # 对标签进行映射
        map = {
            1: 0,  # 弹窗未下单
            2: 0,  # 弹窗三方下单
            0: 1,  # 弹出的券下单
            -1: -1,  # 稍后过滤掉
        }
        label_cast_stage = (
            CastStage(col_name="label_value", map=map, recover_ori_col=True)
            .after(label_data_read_stage)
            .set_input(label_data_read_stage.output_data_names[0])
            .set_pipeline(p)
        )

        final_label_stage = (
            filter_sample_label()
            .after(label_cast_stage)
            .set_input(label_cast_stage.output_data_names[0])
            .set_pipeline(p)
        )

        """-------------获取各类标签-------------"""

        # lucky_member_group_type 客群分类标签
        path = f"/projects/growth/prod/user-reach-label/data/order/order_message_part2/{sample_date}"
        read_data_stage_1 = HDFSORCReadStage(
            path=path, select_cols=["member_id", "lucky_member_group_type"], overwrite=False
        ).set_pipeline(p)

        # join到label上
        join_stage_1 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([final_label_stage.output_data_names[0], read_data_stage_1.output_data_names[0]])
            .after([final_label_stage, read_data_stage_1])
        )

        # is_alt_member 是否小号
        path = f"/projects/growth/prod/user-reach-label/view/label_view/{sample_date}"
        read_data_stage_2 = HDFSORCReadStage(
            path=path, select_cols=["member_id", "is_alt_member"], overwrite=False
        ).set_pipeline(p)

        # join到label上
        join_stage_2 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_1.output_data_names[0], read_data_stage_2.output_data_names[0]])
            .after([join_stage_1, read_data_stage_2])
        )

        # start_days_30d 用户启动天数_30天
        path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/login_order_labels/{sample_date}"
        read_data_stage_3 = HDFSORCReadStage(
            path=path, select_cols=["member_id", "start_days_30d"], overwrite=False
        ).set_pipeline(p)

        # join到label上
        join_stage_3 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_2.output_data_names[0], read_data_stage_3.output_data_names[0]])
            .after([join_stage_2, read_data_stage_3])
        )

        # is_start_no_order_prefer_user is_coupon_list_bw_no_order_prefer_user
        path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/login_order_labels/{sample_date}"
        read_data_stage_4 = HDFSORCReadStage(
            path=path,
            select_cols=[
                "member_id",
                "is_start_no_order_prefer_user",  # 有无启动不下单偏好
                # "is_coupon_list_bw_no_order_prefer_user",  # 有无浏览优惠券列表不下单偏好 没有这个玩意儿
            ],
            overwrite=False,
        ).set_pipeline(p)

        # join到label上
        join_stage_4 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_3.output_data_names[0], read_data_stage_4.output_data_names[0]])
            .after([join_stage_3, read_data_stage_4])
        )

        # psm_360d psm_360天
        path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/member_income_psm_360d/{sample_date.replace('_', '-')}"
        read_data_stage_5 = HDFSORCReadStage(
            path=path, select_cols=["member_id", "psm_360d"], overwrite=False
        ).set_pipeline(p)

        # join到label上
        join_stage_5 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_4.output_data_names[0], read_data_stage_5.output_data_names[0]])
            .after([join_stage_4, read_data_stage_5])
        )

        # app_coupon_list_bw_days_30d 优惠券列表曝光天数_30天
        path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/coupon_bw_labels/{sample_date}"
        read_data_stage_6 = HDFSORCReadStage(
            path=path,
            select_cols=["member_id", "app_coupon_list_bw_days_30d"],
            overwrite=False,
        ).set_pipeline(p)

        # join到label上
        join_stage_6 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_5.output_data_names[0], read_data_stage_6.output_data_names[0]])
            .after([join_stage_5, read_data_stage_6])
        )

        # coupon_used_99_times all_coupon_used_times
        path = f"/projects/growth/prod/user-reach-label/data/coupon-control/base/member_use_99_coupon_180d/{sample_date}"
        read_data_stage_7 = HDFSORCReadStage(
            path=path,
            select_cols=[
                "member_id",
                "coupon_used_99_times",  # 店庆券使用次数_180天
                "all_coupon_used_times",  # 优惠券使用次数_180天
            ],
            overwrite=False,
        ).set_pipeline(p)

        # join到label上
        join_stage_7 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_6.output_data_names[0], read_data_stage_7.output_data_names[0]])
            .after([join_stage_6, read_data_stage_7])
        )

        # active_user_churn_label 流失概率预测等级
        active_user_churn_label_path = f"/projects/growth/prod/wakeup_model/churn_prob/predict_result/{sample_date}/part-0.csv"
        read_data_stage_8 = HDFSCSVReadStage(
            path=active_user_churn_label_path,
            select_cols=["active_user_churn_label"],
            overwrite=False,
        ).set_pipeline(p)

        # join到label上
        join_stage_8 = (
            MultiJoin(on="member_id", how="left")
            .set_pipeline(p)
            .set_inputs([join_stage_7.output_data_names[0], read_data_stage_8.output_data_names[0]])
            .after([join_stage_7, read_data_stage_8])
        )

    p.start(visualize=True, save_path=True, force_rerun=False)

    logging.info("开始运行Polars LazyFrame")
    lz_df_handle = p.get_graph_last_output()
    lz_df_handle.coloct()