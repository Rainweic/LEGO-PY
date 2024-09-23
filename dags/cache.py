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
import sqlite3
import json
import diskcache


class Cache(ABC):
    """
    所有缓存实现的抽象基类。所有子类必须实现read、write和delete方法。
    """

    @abstractmethod
    def read(self, *args, **kwargs): ...

    @abstractmethod
    def write(self, *args, **kwargs): ...

    @abstractmethod
    def delete(self, *args, **kwargs): ...


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
        self.finish_init = False

    def init(self):
        if not self.finish_init:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.cursor.execute(
                """CREATE TABLE IF NOT EXISTS cache
                                (key TEXT PRIMARY KEY, value BLOB)"""
            )
            self.conn.commit()
            self.finish_init = True

    def read(self, k: str) -> bytes:
        """从SQLite给定关联的字符串键读取值。"""
        self.init()
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        self.cursor.execute("SELECT value FROM cache WHERE key=?", (k,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def write(self, k: str, v: bytes) -> None:
        self.init()
        """给定键值对,将值写入SQLite。"""
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        if not isinstance(v, (str, bytes)):
            raise InvalidValueTypeException("请确保值是字符串或字节类型")

        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (k, v)
        )
        self.conn.commit()

    def delete(self, k: str) -> None:
        """给定关联的字符串键,从SQLite删除值。"""
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        self.cursor.execute("DELETE FROM cache WHERE key=?", (k,))
        self.conn.commit()


class DiskCache(Cache):
    """
    为希望继承此功能的用户实现磁盘缓存。我们使用来自pypi的diskcache Python包来实现此缓存功能。
    """

    def __init__(self, disk_cache: diskcache.Cache):
        if not isinstance(disk_cache, diskcache.Cache):
            raise InvalidCacheTypeException("请确保disk_cache的类型是diskcache.Cache")

        self.disk_cache = disk_cache

    def read(self, k: str) -> bytes:
        """给定关联的字符串键,从磁盘缓存读取值。"""
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        return self.disk_cache[k]

    def write(self, k: str, v: bytes) -> None:
        """给定键值对,将值写入磁盘缓存。"""
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        if not isinstance(v, (str, bytes)):
            raise InvalidValueTypeException("请确保值是字符串或字节类型")

        self.disk_cache[k] = v

    def delete(self, k: str) -> None:
        """给定关联的字符串键,从磁盘缓存删除值。"""
        if not isinstance(k, str):
            raise InvalidKeyTypeException("请确保键是字符串")

        self.disk_cache.delete(k)
