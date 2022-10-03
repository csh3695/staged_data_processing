import pickle
import shutil
from abc import ABC, abstractmethod
from inspect import signature
from pathlib import Path
from typing import Callable, Any, Dict, Tuple

import os
import pandas as pd


class Promise:
    def __init__(self):
        self._executed: bool = False
        self._result: Any = None
        self._fn: Callable = None
        self._args: Tuple[Any] = None
        self._kwargs: Dict[str, Any] = None

    def bind_function(
        self, fn: Callable[..., Any], args: Tuple[Any], kwargs: Dict[str, Any]
    ):
        assert callable(fn), "fn must be callable"
        self._fn = fn
        self._args = args or tuple()
        self._kwargs = kwargs or dict()
        return self

    def execute(self):
        if self._fn is None:
            raise ValueError("No function bound to promise")
        if not self._executed:
            self._result = self._fn(
                *[
                    arg.execute() if isinstance(arg, Promise) else arg
                    for arg in self._args
                ],
                **{
                    k: v.execute() if isinstance(v, Promise) else v
                    for k, v in self._kwargs.items()
                },
            )
            self._executed = True
        return self._result

    def _set_result(self, result):
        self._result = result
        self._executed = True

    def __repr__(self):
        if self._fn is None:
            return f"{self.__class__.__name__}()"
        inner_repr = f"{self._fn.__name__}{signature(self._fn).bind(*self._args, **self._kwargs).args}".split(
            "\n"
        )
        inner_repr = "\n\t".join(inner_repr)
        return f"{self.__class__.__name__}(\n\t{inner_repr}\n)"


class CachePromise(Promise, ABC):
    def __init__(self, cache_dir: str):
        super().__init__()
        self.cache_dir = cache_dir

    @abstractmethod
    def load(self, cache_dir: str):
        """
        Load the result from cache. Should return None if cache is not available, not raising Exceptions.
        """
        pass

    @abstractmethod
    def save(self, cache_dir: str):
        """
        Save the result to cache.
        """
        pass

    def execute(self):
        if not self._executed:
            result = self.load(self.cache_dir)
            if result is not None:
                self._set_result(result)
            else:
                self._result = self._fn(
                    *[
                        arg.execute() if isinstance(arg, Promise) else arg
                        for arg in self._args
                    ],
                    **{
                        k: v.execute() if isinstance(v, Promise) else v
                        for k, v in self._kwargs.items()
                    },
                )
                self._executed = True
                self.save(self.cache_dir)
        return self._result

    def __repr__(self):
        return super().__repr__() + f" [cache_dir={self.cache_dir}]"


class SparkParquetCachePromise(CachePromise):
    def __init__(self, cache_dir: str, spark_session):
        super().__init__(str(cache_dir))
        self.spark_session = spark_session

    def load(self, cache_dir: str):
        try:
            cached = self.spark_session.read.parquet(cache_dir)
        except Exception as e:
            print(e)
            return None
        if cached.count() == 0:
            return None
        return cached

    def save(self, cache_dir: str):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Write {cache_dir}/*.parquet to disk...")
        self._result.write.parquet(cache_dir, mode="overwrite")

    def __deepcopy__(self, memodict={}):
        return SparkParquetCachePromise(self.cache_dir, self.spark_session)

    def __repr__(self):
        return super().__repr__() + f" [spark_session={self.spark_session}]"


class PickleCachePromise(CachePromise):
    def __init__(self, cache_dir: str):
        super().__init__(str(cache_dir))

    def _get_cache_file_name(self, cache_dir: str):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(cache_dir, f"{self._fn.__name__}.cache.pkl")

    def load(self, cache_dir: str):
        try:
            with open(self._get_cache_file_name(cache_dir), "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(e)
            return None

    def save(self, cache_dir: str):
        dest = self._get_cache_file_name(cache_dir)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"Write {dest} to disk...")
        with open(dest, "wb") as f:
            pickle.dump(self._result, f)


class PandasParquetCachePromise(CachePromise):
    def __init__(self, cache_dir: str):
        super().__init__(str(cache_dir))

    def _get_cache_file_name(self, cache_dir: str):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(cache_dir, f"{self._fn.__name__}.cache.parquet")

    def load(self, cache_dir: str):
        try:
            cached = pd.read_parquet(self._get_cache_file_name(cache_dir))
        except Exception as e:
            print(e)
            return None
        if cached.shape[0] == 0:
            return None
        return cached

    def save(self, cache_dir: str):
        dest = self._get_cache_file_name(cache_dir)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"Write {dest} to disk...")
        self._result.to_parquet(dest, index=False)
