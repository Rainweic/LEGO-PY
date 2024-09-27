主要参考`demo1.yaml`、`demo2.yaml`

以下是参数说明

```yaml

# 全局参数
global_args:

  args_1: hello world
  args_2: 2024-09-05

pipeline:
    # 以下构建了两个pipeline：A、B
    - name: A                           # pipeline命名
        
        # 当前pipeline参数
        args:
            visualize: True             # 是否生成可视化DAG图
            save_dags: True             # 是否保存DAG图
            force_rerun: False          # 强制重跑（运行中失败会中断续跑，通过该参数重跑）

        
        stage:
            - CustomFunc:                                       # 自定义函数类型（自定义类则使用CustomStage）
                                                                # 支持的类型查看 stages 下的所有类

                path: .                                         # .代表函数内容存放在和当前yaml文件同名同路径下的py文件内
                module_name: func_name                          # 自定义函数名
                name: final_label_stage                         # stage命名
                after:                                          # 上一个stage的名称
                    - pre_stage_name                            
                inputs:                                         # 指定输入
                    - pre_stage_name.1                          # .n代表某个stage的第n个输出
                args:                                           # 参数设定
                    - a: "${args_1}"
                    - b: "${args_2}"
                # 若计算数据为polars.LazyFrame, 该参数则代表是否在组件计算完成后是否执行collect()函数
                # LazyFrame doc: https://docs.pola.rs/user-guide/lazy/using/#using-the-lazy-api-from-a-file
                # collect() doc: https://docs.pola.rs/user-guide/lazy/execution/#execution-on-larger-than-memory-data
                collect_result: False
    - name: B
        ...
```