
## Pipeline的搭建和运行


### Pipeline搭建


1. 定义各个处理阶段（Stage）：

    - 每个Stage都是`CustomStage`的子类或使用`@stage`装饰器定义的函数。
    - demo1.py: 使用`@stage`装饰器来对函数进行包装
    - demo2.py: 继承`CustomStage`来对复杂功能进行包装
    - 两个可以混合使用

2. 自定义`build_pipeline()`函数用来构建pipeline：

    - 创建各个Stage的实例。【参考demo1.py demo2.py】
    - 使用`.after()`方法设置Stage之间的依赖关系。
    - 使用`.set_input()`/`.set_inputs()`设置输入。
    - 每个Stage都需要设置输出数量。设置方法参考demo
    - 创建`Pipeline`实例并添加所有Stage。


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