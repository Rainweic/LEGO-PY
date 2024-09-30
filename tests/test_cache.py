import unittest
import aiounittest
from dags.cache import SQLiteCache, InvalidKeyTypeException, InvalidValueTypeException
import os
import tempfile

class TestSQLiteCache(aiounittest.AsyncTestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.db_path = tempfile.mktemp(suffix=".db")
        self.cache = SQLiteCache(db_path=self.db_path)

    async def test_write_and_read(self):
        await self.cache.write("test_key", b"test_value")
        result = await self.cache.read("test_key")
        self.assertEqual(result, b"test_value")

    async def test_delete(self):
        await self.cache.write("test_key", b"test_value")
        await self.cache.delete("test_key")
        result = await self.cache.read("test_key")
        self.assertIsNone(result)

    async def test_invalid_key_type(self):
        with self.assertRaises(InvalidKeyTypeException):
            await self.cache.write(123, b"test_value")

    async def test_invalid_value_type(self):
        with self.assertRaises(InvalidValueTypeException):
            await self.cache.write("test_key", 123)


if __name__ == "__main__":
    unittest.main()