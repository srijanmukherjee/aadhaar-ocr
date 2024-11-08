from contextlib import contextmanager
from time import perf_counter

@contextmanager
def measure_time():
    t1 = t2 = perf_counter() 
    yield lambda: t2 - t1
    t2 = perf_counter()
