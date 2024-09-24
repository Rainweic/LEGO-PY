import polars as pl
from stages.where import Where

# 创建示例数据
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
}

# 转换为 LazyFrame
df = pl.DataFrame(data).lazy()

# 示例 1: 根据单个条件过滤
where_stage = Where("age > 30")
filtered_df = where_stage.forward(df).collect()
print(filtered_df)
# 断言结果
expected_data_1 = {
    "name": ["Charlie", "David", "Eve"],
    "age": [35, 40, 45],
    "city": ["Chicago", "Houston", "Phoenix"]
}
expected_df_1 = pl.DataFrame(expected_data_1)
assert filtered_df.equals(expected_df_1), "Test 1 failed"

# 示例 2: 根据多个条件过滤
where_stage = Where("age > 30", "city = 'Chicago'")
filtered_df = where_stage.forward(df).collect()
print(filtered_df)
# 断言结果
expected_data_2 = {
    "name": ["Charlie"],
    "age": [35],
    "city": ["Chicago"]
}
expected_df_2 = pl.DataFrame(expected_data_2)
assert filtered_df.equals(expected_df_2), "Test 2 failed"
