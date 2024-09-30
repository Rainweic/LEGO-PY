import polars as pl
import pickle
import diskcache
import time
import os

def generate_large_lazyframe(rows, cols):
    """生成一个大型的LazyFrame，大约5GB大小。"""
    data = {f"col_{i}": pl.Series(range(rows)) for i in range(cols)}
    lf = pl.DataFrame(data).lazy()
    return lf

def test_pickle(lf, filename):
    """使用pickle序列化和反序列化LazyFrame，并测量时间。"""
    start_time = time.time()
    with open(filename, 'wb') as f:
        pickle.dump(lf, f)
    print(f"Pickle Write Time: {time.time() - start_time} seconds")

    start_time = time.time()
    with open(filename, 'rb') as f:
        _ = pickle.load(f)
    print(f"Pickle Read Time: {time.time() - start_time} seconds")

def test_diskcache(lf, cache_dir):
    """使用diskcache序列化和反序列化LazyFrame，并测量时间。"""
    cache = diskcache.Cache(cache_dir)
    start_time = time.time()
    cache['lf'] = lf
    print(f"DiskCache Write Time: {time.time() - start_time} seconds")

    start_time = time.time()
    _ = cache['lf']
    print(f"DiskCache Read Time: {time.time() - start_time} seconds")
    cache.close()

def main():
    rows = 5000000  # 大约5GB数据
    cols = 10
    lf = generate_large_lazyframe(rows, cols)

    # 测试pickle
    test_pickle(lf, 'large_lf.pkl')

    # 测试diskcache
    test_diskcache(lf, 'diskcache_dir')

if __name__ == "__main__":
    main()