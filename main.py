import numpy as np
import pandas as pd

from staged import stage
from staged.promise import PickleCachePromise, PandasParquetCachePromise


def task_0():
    print("task_0")
    return 10


@stage()
def task_1():
    print("task_1")
    return 11


@stage(PickleCachePromise(".cache/task_2/"))
def task_2(x: int, y: int):
    print("task_2")
    return list(range(x, (max(x, 0) + 10) * 2)) + list(range(y, (max(y, 0) + 2) * 2))


@stage(PickleCachePromise(".cache/task_3/"))
def task_3(x: list):
    print("task_3")
    return np.array(x)


@stage(PandasParquetCachePromise(".cache/task_4/"))
def task_4(x: np.ndarray, y: list, z: np.ndarray):
    print("task_4")
    minlen = min(len(x), len(y), len(z))
    return pd.DataFrame({"x": x[:minlen], "y": y[:minlen], "z": z[:minlen]})


@stage(PandasParquetCachePromise(".cache/task_5/"))
def task_5(x: pd.DataFrame):
    print("task_5")
    x["x2"] = x["x"] * 2
    x["y2"] = x["y"] * 2
    x["z2"] = x["z"] * 2
    return x


r0 = task_0()
r1 = task_1()
r1_ = [
    task_1() for _ in range(100)
]  # 0 execution because execution is not propagated from the final result r5(Dangling Promise).
r2 = task_2(r0, r1)
r3 = task_3(r2)
r4 = task_4(r3, r2, r3)  # r3 is referenced twice but only executed once
r5 = task_5(r4)
r5.execute()
