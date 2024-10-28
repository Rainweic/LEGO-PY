import polars as pl
import unittest
from stages.sample import Sample

class TestSample(unittest.TestCase):

    def test_random_sampling(self):
        df = pl.DataFrame({
            "a": range(10)
        })
        sampler = Sample(n_sample=5, random=True, seed=100)
        result = sampler.forward(df).collect()
        print(result)
        self.assertEqual(result.height, 5)  # 验证采样后的行数是否正确

    def test_sequential_sampling(self):
        df = pl.DataFrame({
            "a": range(10)
        })
        sampler = Sample(n_sample=5, random=False)
        result = sampler.forward(df).collect()
        self.assertEqual(result.shape[0], 5)  # 验证采样后的行数是否正确
        self.assertEqual(result.to_pandas().to_dict(), {'a': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}})  # 验证是否为顺序采样

if __name__ == "__main__":
    unittest.main()