

### 现有功能

- DataFrame依赖Polars Lazy API，延迟计算，支持小内存大数据
- 支持中断续跑
- 组件输出临时存储
- 流程图输出
- 自定义各类功能组件
- 读取yaml配置文件自动生成计算图 [参考demo1.yaml、demo2.yaml]
- 功能排期：http://wiki.lkcoffee.com/x/c1iGE


### 准备工作

clone项目
```bash
git clone http://git.lkcoffee.com/luqin.li/py-lego.git
cd py-lego
```
安装依赖
```bash
pip install -r ./requirements.txt
```

### 开启后端【需配合前端使用】

```bash
sh deploy_local.sh
```

### 本地运行demo【用于开发测试】

方式一：通过代码构建的计算图直接运行
```bash
python demo1.py
python demo2.py
```

方式二：通过写好的yaml文件生成计算图并运行
```bash
python parser_yaml.py -p ./demo1.yaml
python parser_yaml.py -p ./demo2.yaml
```

### 教程相关


1. 定义各个处理阶段（Stage）：

    - 每个Stage都是`CustomStage`的子类或使用`@stage`装饰器定义的函数。
    - demo1.py: 使用`@stage`装饰器来对函数进行包装
    - demo2.py: 继承`CustomStage`来对复杂功能进行包装
    - 两个可以混合使用

    [具体教程](./docs/Add%20stage.md)

2. 自定义`build_pipeline()`函数用来构建pipeline：

    - 使用`Pipeline`创建计算图，【参考demo1.py demo2.py】
        - 使用`.set_pipeline()`方法设置指定Pipeline
        - 使用`.after()`方法设置Stage之间的依赖关系。
        - 使用`.set_input()`/`.set_inputs()`设置输入。
    - 使用`Pipeline.get_output`获取指定stage的输出。
    - 每个Stage都需要设置输出数量。设置方法参考demo

3. [yaml文件说明](./docs/Yaml%20file%20describe.md)



### BUG && TODO

P0
- ✅运行全部画布好像少了一个节点？？？
- ✅某些节点运行错误会导致别的节点无法设置状态???

P1
- 运行节点后设置状态为waiting，随后在running

P3
- IO加速
- 直接从hdfs用polars进行读取数据
