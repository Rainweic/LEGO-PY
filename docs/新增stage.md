新增一个计算组件（stage）有两种方法：

- 使用@stage装饰器装饰函数
- 继承BaseStage

输入输出建议都使用Polars的LazyFrame

1. 使用@stage装饰器装饰函数，使用例子如下：

```python
from dags.stage import stage
from dags.pipeline import Pipeline

# 定义stage
@stage(n_outputs=1)                                         # n_outputs设置组件输出数量
def pre_stage(name):                                        # name参数等会儿从外部传入
    print("pre_stage")
    print(f"Hello {name}!")                         
    return 10, 20


# 定义stage
@stage(n_outputs=0)                                         # n_outputs设置组件输出数量
def next_stage(pre_out_0, pre_out_1, custom_arg):           # 前几个参数为上一个组件的输出数据
    print(f"输入数据：{pre_out_0}, {pre_out_1}")
    print(f"自定义传入参数：{custom_arg}")

def run():
    with Pipeline() as p:
        
        pre_s = pre_stage(name="World")\                    # 自定义参数通过这种方式传入
            .set_pipeline(p)

        next_s = next_stage(custom_arg="This is next stage")\
            .set_pipeline(p)\
            .after(pre_s)\
            .set_inputs(pre_s.output_data_names)

run()

```

2. 继承BaseStage，使用例子参考demo2.py

3. **新增常用组件请在文件夹 `./stages` 下进行添加**


