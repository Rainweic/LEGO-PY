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
    # raise RuntimeError
    return c * 3


@stage(n_outputs=2)
def stage_4(d):
    return d, 4


@stage(n_outputs=1)
def stage_5(e):
    return e**5


@stage(n_outputs=0)
def stage_6(f, g, h, i):
    print(f / g * h + i)


with Pipeline() as p:

    stage1 = stage_1().set_pipeline(p)

    stage2 = stage_2().set_pipeline(p)

    stage3 = (
        stage_3().set_pipeline(p).after(stage2).set_input(stage2.output_data_names[0])
    )

    stage4 = (
        stage_4().set_pipeline(p).after(stage2).set_input(stage2.output_data_names[0])
    )

    stage5 = (
        stage_5().set_pipeline(p).after(stage1).set_inputs(stage1.output_data_names)
    )

    stage6 = (
        stage_6()
        .set_pipeline(p)
        .after([stage3, stage4, stage5])
        .set_inputs(
            stage3.output_data_names
            + stage4.output_data_names
            + stage5.output_data_names
        )
    )

p.start(visualize=True, save_path=True)
