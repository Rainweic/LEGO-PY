import pickle
import polars as pl
from dags.cache import SQLiteCache


async def load_output(job_id: str, stage_name: str, output_idx: str):
    
    try:
        cache = SQLiteCache()
        output_names = await cache.read(f"{job_id}_{stage_name}_output_names")
        output_name = pickle.loads(output_names)[output_idx]
        data_path = await cache.read(output_name)
    except Exception as e:
        error_msg = f"读取数据失败: {e}"
        raise IOError(error_msg)
    
    if data_path.endswith('.parquet'):
        return pl.read_parquet(data_path)
    elif data_path.endswith('.pickle'):
        with open(data_path, 'rb') as f:    
            return pickle.load(f)
    else:
        error_msg = f"不支持的文件格式: {data_path}"
        raise TypeError(error_msg)
