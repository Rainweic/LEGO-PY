
## Pipeline的搭建和运行


### 现有功能

- DataFrame依赖Polars Lazy API，延迟计算，支持小内存大数据
- 支持中断续跑
- 组件输出临时存储
- 流程图输出
- 自定义各类功能组件
- 读取yaml配置文件自动生成计算图 [参考demo1.yaml、demo2.yaml]
- 支持组件：
    - hdfs数据下载
    - join
    - label_enconder
    - pearson 
    - spearman
    - where


### Pipeline搭建


1. 定义各个处理阶段（Stage）：

    - 每个Stage都是`CustomStage`的子类或使用`@stage`装饰器定义的函数。
    - demo1.py: 使用`@stage`装饰器来对函数进行包装
    - demo2.py: 继承`CustomStage`来对复杂功能进行包装
    - 两个可以混合使用

2. 自定义`build_pipeline()`函数用来构建pipeline：

    - 使用`Pipeline`创建计算图，【参考demo1.py demo2.py】
        - 使用`.set_pipeline()`方法设置指定Pipeline
        - 使用`.after()`方法设置Stage之间的依赖关系。
        - 使用`.set_input()`/`.set_inputs()`设置输入。
    - 使用`Pipeline.get_output`获取指定stage的输出。
    - 每个Stage都需要设置输出数量。设置方法参考demo



### 未实现

- 前端拖拉拽UI，生成配置文件
- 异步读写输入输出
- 上下两个组件的输入输出性能优化
