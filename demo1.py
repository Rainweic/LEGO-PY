from dags.pipeline import Pipeline
from dags.stage import stage


@stage(n_outputs=1)
def stage_1():
    return 1

@stage(n_outputs=1)
def stage_2():
    return 1

@stage(n_outputs=1)
def stage_3(c):
    return c * 3

@stage(n_outputs=2)
def stage_4(d):
    return d, 4

@stage(n_outputs=1)
def stage_5(e):
    return e ** 5

@stage(n_outputs=0)
def stage_6(f, g, h, i):
    print(f / g * h + i)

def build_pipeline():

    stage1 = stage_1()

    stage2 = stage_2()

    stage3 = stage_3() \
        .after(stage2) \
        .set_input(stage2.output_data_names[0])

    stage4 = stage_4() \
        .after(stage2) \
        .set_input(stage2.output_data_names[0])
    
    stage5 = stage_5() \
        .after(stage1) \
        .set_inputs(stage1.output_data_names)
    
    stage6 = stage_6() \
        .after([stage3, stage4, stage5]) \
        .set_inputs(stage3.output_data_names + stage4.output_data_names + stage5.output_data_names)

    pipeline = Pipeline()

    pipeline.add_stages([
        stage1, stage2, stage3,
        stage4, stage5, stage6
    ])
    
    return pipeline


pipeline = build_pipeline()

pipeline.start(visualize=True)