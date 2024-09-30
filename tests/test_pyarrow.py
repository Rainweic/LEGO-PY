import pyarrow as pa
import pickle
import time
import numpy as np

def serialize_with_arrow(data):
    """使用 PyArrow 序列化数据"""
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, data.schema)
    writer.write_table(data)
    writer.close()
    return sink.getvalue()

def deserialize_with_arrow(buffer):
    """使用 PyArrow 反序列化数据"""
    reader = pa.ipc.open_stream(buffer)
    return reader.read_all()

# 生成大型数据
num_features = 100
num_records = 50000000
data = {f"col_{i}": np.random.rand(num_records) for i in range(num_features)}
table = pa.table(data)

# 使用 PyArrow 序列化
start_time = time.time()
serialized_data_arrow = serialize_with_arrow(table)
arrow_serialize_time = time.time() - start_time

# 使用 PyArrow 反序列化
start_time = time.time()
deserialized_data_arrow = deserialize_with_arrow(serialized_data_arrow)
arrow_deserialize_time = time.time() - start_time

# 使用 pickle 序列化
start_time = time.time()
serialized_data_pickle = pickle.dumps(table)
pickle_serialize_time = time.time() - start_time

# 使用 pickle 反序列化
start_time = time.time()
deserialized_data_pickle = pickle.loads(serialized_data_pickle)
pickle_deserialize_time = time.time() - start_time

print(f"PyArrow Serialize Time: {arrow_serialize_time:.2f} seconds")
print(f"PyArrow Deserialize Time: {arrow_deserialize_time:.2f} seconds")
print(f"Pickle Serialize Time: {pickle_serialize_time:.2f} seconds")
print(f"Pickle Deserialize Time: {pickle_deserialize_time:.2f} seconds")