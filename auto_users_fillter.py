import pandas as pd

from dags.stage import stage
from dags.pipeline import Pipeline
from stages import *


@stage
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

    # 读取标签数据
    label_data_path = f"/projects/growth/test/analysis_tools/alg_label/coupon_planned_strategy/expose_label/single/{sample_date}"
    label_data_read_stage = HDFSCSVReadStage(path=label_data_path, overwrite=False) \
        .set_default_outputs(n_outputs=1)

    # 对标签进行映射
    map = {
        1: 0,   # 弹窗未下单
        2: 0,   # 弹窗三方下单
        0: 1,   # 弹出的券下单
        -1: -1  # 稍后过滤掉
    }
    label_cast_stage = CastStage(feature_name="label_value", map=map) \
                            .after(label_data_read_stage) \
                            .set_input(label_data_read_stage.output_data_names[0]) \
                            .set_default_outputs(n_outputs=1)

    
    final_label_stage = filter_sample_label() \
        .after(label_cast_stage) \
        .set_input(label_cast_stage.output_data_names[0]) \
        .set_default_outputs(n_outputs=1)
    
    pipeline = Pipeline()
    pipeline.add_stages([
        label_data_read_stage,
        label_cast_stage,
        final_label_stage
    ])

    return pipeline


# 运行
if __name__ == "__main__":
    sample_date = "2024-09-05"
    pipeline = build_pipeline()
    pipeline.start()
