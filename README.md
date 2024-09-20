
## Pipeline的搭建和运行


### Pipeline搭建


1. 定义各个处理阶段（Stage）：

    - 每个Stage都是`BaseStage`的子类或使用`@stage`装饰器定义的函数。
    - demo1.py: 使用`@stage`装饰器来对函数进行包装
    - demo2.py: 继承`BaseStage`来对复杂功能进行包装
    - 两个可以混合使用

2. 自定义`build_pipeline()`函数用来构建pipeline：

    - 创建各个Stage的实例。【参考demo1.py demo2.py】
    - 使用`.after()`方法设置Stage之间的依赖关系。
    - 使用`.set_input()`/`.set_inputs()`和`.set_default_outputs()`方法设置输入输出。
    - 创建`Pipeline`实例并添加所有Stage。
    - 示例代码：
    ```python
    def build_pipeline():
        stage1 = Stage1().set_default_outputs(n_outputs=1)
        stage2 = Stage2().after(stage1).set_inputs(stage1.output_data_names).set_default_outputs(n_outputs=1)
        stage3 = Stage3().after(stage2).set_inputs(stage2.output_data_names)

        pipeline = Pipeline()
        pipeline.add_stages([stage1, stage2, stage3])
        return pipeline
    ```


### Pipeline运行

1. 构建pipeline：
   ```python
   pipeline = build_pipeline()
   ```

2. 启动pipeline：
   ```python
   pipeline.start()
   ```

可以在`start()`方法中添加参数，如`visualize=True`来可视化pipeline结构。