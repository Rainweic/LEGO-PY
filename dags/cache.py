"""
此模块包含用于DAG执行期间的管道内部以及用户可能子类化的缓存的基类和实现。
任何缓存都必须至少实现'read'、'write'和'delete'方法。

这里实现了两种缓存:由SQLite支持的本地数据库缓存和本地磁盘缓存。

SQLite缓存用于管道读/写与管道及其组成阶段执行相关的数据(例如,正在进行中、已完成的阶段等)。

磁盘缓存不被管道使用,但它与SQLite缓存一样,可以被用户子类化以继承相关的缓存功能。
这样做最常见的用例是在DAG中传递数据。

参考:
    https://pypi.org/project/diskcache/
    https://docs.python.org/3/library/sqlite3.html
"""

from abc import ABC, abstractmethod
import aiosqlite
import asyncio


class Cache(ABC):
    """
    所有缓存实现的抽象基类。所有子类必须实现read、write和delete方法。
    """

    @abstractmethod
    async def read(self, *args, **kwargs): ...

    @abstractmethod
    async def write(self, *args, **kwargs): ...

    @abstractmethod
    async def delete(self, *args, **kwargs): ...


class InvalidCacheTypeException(Exception):
    pass


class InvalidKeyTypeException(Exception):
    pass


class InvalidValueTypeException(Exception):
    pass


class SQLiteCache(Cache):
    """
    实现一个本地数据库缓存,供Pipeline实现本身使用,也可能供希望继承此功能的用户使用。
    使用的本地数据库技术是SQLite。

    使用此缓存假定已安装sqlite3模块(通常随Python一起提供)。
    """

    def __init__(self, db_path: str = "./pipeline.db"):
        self.db_path = db_path
        self.conn = None

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)""")
            await db.commit()

    async def read(self, k: str) -> bytes:
        """从SQLite给定关联的字符串键读取值。"""
        await self.init()
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")
        
        async with aiosqlite.connect(self.db_path) as db:
            result = await db.execute("SELECT value FROM cache WHERE key=?", (k,))
            result = await result.fetchone()
            return result[0] if result else None

    async def write(self, k: str, v: bytes) -> None:
        """给定键值对,将值写入SQLite。"""
        await self.init()
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        if not isinstance(v, (str, bytes)):
            raise InvalidValueTypeException("请确保值是字符串或字节类型")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (k, v))
            await db.commit()

    def write_sync(self, k: str, v: bytes) -> None:
        """给定键值对,将值写入SQLite的同步版本。"""
        import sqlite3
        
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        if not isinstance(v, (str, bytes)):
            raise InvalidValueTypeException(f"请确保值是字符串或字节类型: {v}")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (k, v))
            conn.commit()

    async def delete(self, k: str) -> None:
        """给定关联的字符串键,从SQLite删除值。"""
        await self.init()
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cache WHERE key=?", (k,))
            await db.commit()
