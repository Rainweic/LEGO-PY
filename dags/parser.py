import yaml
import asyncio
import os
import importlib.util
import traceback
from utils.logger import setup_logger
from stages import create_stage
from dags.pipeline import Pipeline, StageStatus


def import_module_from_script(script_path: str, module_name: str):
    """
    从指定的脚本路径中导入指定的函数/类。

    参数:
        script_path (str): 脚本文件的路径
        module_name (str): 要导入的函数名/类名

    返回:
        函数对象
    """
    # 确保脚本文件存在
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"脚本文件未找到: {script_path}")

    # 动态导入模块
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取指定的函数
    module = getattr(module, module_name, None)
    if module is None:
        raise AttributeError(f"模块未找到: {module_name}")

    return module


async def load_pipelines_from_yaml(yaml_file: str) -> list[Pipeline]:

    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # print(config)
        
    # 解析全局参数
    global_args = config.get('global_args', {})

    out_p = []
    pipelines_config = config.get('pipeline', [])

    for pipeline_config in pipelines_config:

        job_id = pipeline_config.get("name", None)

        # logging = setup_logger(name="parser", prefix="parser", job_id=job_id)

        pipeline_args = pipeline_config.get("args", {})
        parallel = pipeline_args.get("parallel", None)
        visualize = pipeline_args.get("visualize", False)
        save_dags = pipeline_args.get("save_dags", True)
        force_rerun = pipeline_args.get("force_rerun", False)  

        async with Pipeline(parallel=parallel, visualize=visualize, save_dags=save_dags, force_rerun=force_rerun, job_id=job_id) as p:

            instance = {}
            # print(pipeline_config['stages'])

            # 解析各个阶段
            for stage in pipeline_config.get('stages', []):

                p.logger.info(f"Auto create stage...: {stage}")
                # print(pipeline_config["stages"])

                stage_type = list(stage.keys())[0]
                stage_info = stage[stage_type]

                # 替换全局参数
                for key, value in stage_info.get('args', {}).items():
                    if isinstance(value, str) and '${global_args.' in value:
                        s_begin = value.find('${global_args.')
                        s_end = value.find('}')
                        arg_name = value[s_begin + len('${global_args.'): s_end]  # 获取参数名
                        new_value = value[:s_begin] + f"{global_args.get(arg_name, value)}" + value[s_end+1:]
                        stage_info['args'][key] = new_value

                # 创建阶段实例

                if stage_type in ['CustomFunc', 'CustomStage']:
                    try:
                        script_path = stage_info["path"]
                    except KeyError as e:
                        p.logger.error(f"'CustomFunc', 'CustomStage'需要通过path来设定脚本所在路径, '.'代表yaml文件同路径、同名.py文件")
                    if script_path == ".":
                        # 获取yaml文件所在路径
                        script_path = yaml_file.replace(".yaml", ".py")
                    module_name = stage_info["module_name"]
                    # 从script_path中导入指定函数
                    module = import_module_from_script(script_path=script_path, module_name=module_name)
                    stage_args = stage_info.get("args", {})
                    p.logger.info(f"[{stage_type}] {module_name}'s args: {stage_args}")
                    try:
                        stage_instance = module(**stage_args)
                    except Exception as e:
                        error_msg = f"无法创建实例. Stage信息：{stage}"
                        p.logger.error(error_msg)
                        p.logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
                        await p._update_stage_status(stage_info['name'], StageStatus.FAILED)
                        # raise TypeError(error_msg)
                        continue
                    # 设置name
                    name = stage_info['name']
                    if name:
                        stage_instance.name = name
                else:
                    try:
                        stage_instance = create_stage(stage_type, stage_info['name'], stage_info['args'], job_id=job_id)
                    except Exception as e:
                        error_msg = f"输入参数错误：{stage_info['args']}, 无法创建实例. Stage信息：{stage}"
                        p.logger.error(error_msg)
                        p.logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
                        await p._update_stage_status(stage_info['name'], StageStatus.FAILED)
                        # raise TypeError(error_msg)
                        continue
                
                # stage 设置
                stage_instance.set_pipeline(p)
                stage_instance.logger.info(stage);

                if stage_info.get('inputs', None):
                    for i in stage_info["inputs"]:
                        if "." in i:
                            before_instance_name, output_idx = i.split(".")[0], int(i.split(".")[1])

                            if before_instance_name in instance:
                                stage_instance.add_input(instance[before_instance_name].output_data_names[output_idx])
                            else:
                                # 若从某个节点开始执行，上一个节点是没有初始化的 直接拼接
                                stage_instance.add_input(f"{before_instance_name}_output_{output_idx}")
                        else:
                            before_instance_name = i
                            for pre_stage_output_data_name in instance[before_instance_name].output_data_names:
                                stage_instance.add_input(pre_stage_output_data_name)

                if stage_info.get('after', None):

                    before_stages_name = stage_info['after']
                    
                    try:
                        if isinstance(before_stages_name, list):
                            before_stages = [instance[name] for name in before_stages_name]
                        elif isinstance(before_stages_name, str):
                            before_stages = [instance[before_instance_name]]
                        else:
                            raise TypeError("stages.name 必须是字符串或列表")
                        stage_instance.after(before_stages)
                    except KeyError as e:
                        # 这里可能是因为画布从中间节点开始执行，然后获取不到上游的stage，但不影响运行
                        p.logger.warning(f"找不到命名为{e.args[0]}的stage")

                if stage_info.get("collect_result", False):
                    show = stage_info.get("show_collect_result", True)
                    stage_instance.collect_result(show=show)

                instance[stage_info['name']] = stage_instance

                p.logger.info("Finish create stage!")

            out_p.append(p)

    return out_p


# test
if __name__ == "__main__":
    yaml_file_path = "./demo1.yaml"  # YAML文件路径
    asyncio.run(load_pipelines_from_yaml(yaml_file_path))
