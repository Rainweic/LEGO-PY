# 单元测试
import unittest
import polars as pl
from stages.sample import sample_lazyframe

class TestSampleLazyFrame(unittest.TestCase):
    def setUp(self):
        # 创建一个测试用的 LazyFrame
        self.test_df = pl.DataFrame({
            'A': range(100),
            'B': ['x' if i % 2 == 0 else 'y' for i in range(100)]
        }).lazy()

    def test_sample_size(self):
        # 测试采样大小是否正确
        n = 10
        sampled = sample_lazyframe(self.test_df, n).collect()
        self.assertEqual(len(sampled), n)

    def test_seed_reproducibility(self):
        # 测试使用相同的种子是否产生相同的结果
        seed = 42
        sample1 = sample_lazyframe(self.test_df, 10, seed).collect()
        sample2 = sample_lazyframe(self.test_df, 10, seed).collect()
        self.assertTrue(sample1.frame_equal(sample2))

    def test_different_seeds(self):
        # 测试不同的种子是否产生不同的结果
        sample1 = sample_lazyframe(self.test_df, 10, seed=1).collect()
        sample2 = sample_lazyframe(self.test_df, 10, seed=2).collect()
        self.assertFalse(sample1.frame_equal(sample2))

    def test_no_duplicate_rows(self):
        # 测试采样结果中是否没有重复行
        sampled = sample_lazyframe(self.test_df, 50).collect()
        self.assertEqual(len(sampled), len(sampled.unique()))

    def test_input_types(self):
        # 测试函数是否能正确处理 DataFrame 和 LazyFrame 输入
        df = self.test_df.collect()
        sampled_from_df = sample_lazyframe(df, 10).collect()
        sampled_from_lf = sample_lazyframe(self.test_df, 10).collect()
        self.assertEqual(len(sampled_from_df), 10)
        self.assertEqual(len(sampled_from_lf), 10)

if __name__ == '__main__':
    unittest.main()