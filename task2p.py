"""
Task 2 - Pseudo-Random Generator State Bottleneck
Student ID: 230103182

Monte Carlo pi with 40,000,000 points, 4 threads:
  1. one shared PRNG guarded by a mutex (threads take turns generating)
  2. one independent PRNG per thread, seed = Student_ID * 10 + thread_index
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import math
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103182
TOTAL_POINTS = 40_000_000
THREADS = 4
CHUNK = 250_000                      # points generated per request
PER_THREAD = TOTAL_POINTS // THREADS


def count_inside(x: np.ndarray, y: np.ndarray) -> int:
    """Points with x^2 + y^2 <= 1 fall inside the quarter circle."""
    return int(np.count_nonzero(x * x + y * y <= 1.0))


# ---------- Mode 1: shared PRNG + mutex ----------
def run_shared() -> tuple[float, float]:
    rng = np.random.default_rng(STUDENT_ID)
    lock = threading.Lock()

    def worker(_index: int) -> int:
        inside, done = 0, 0
        while done < PER_THREAD:
            n = min(CHUNK, PER_THREAD - done)
            with lock:                          # only one thread may touch the generator state
                x = rng.random(n)
                y = rng.random(n)
            inside += count_inside(x, y)        # counting itself runs in parallel
            done += n
        return inside

    return _launch(worker)


# ---------- Mode 2: independent PRNG per thread ----------
def run_independent() -> tuple[float, float]:
    def worker(index: int) -> int:
        rng = np.random.default_rng(STUDENT_ID * 10 + index)   # private state, no lock
        inside, done = 0, 0
        while done < PER_THREAD:
            n = min(CHUNK, PER_THREAD - done)
            x = rng.random(n)
            y = rng.random(n)
            inside += count_inside(x, y)
            done += n
        return inside

    return _launch(worker)


def _launch(worker) -> tuple[float, float]:
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        t0 = time.perf_counter()
        inside = sum(pool.map(worker, range(THREADS)))
        elapsed_ms = (time.perf_counter() - t0) * 1000
    pi = 4.0 * inside / (PER_THREAD * THREADS)
    return pi, elapsed_ms


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")
    print(f"Samples: {TOTAL_POINTS:,} | threads: {THREADS} | chunk: {CHUNK:,} points per request")
    print(f"Independent seeds: {[STUDENT_ID * 10 + i for i in range(THREADS)]}")
    print()

    run_independent()                           # warm-up (allocations, page faults)

    pi_s, t_s = run_shared()
    pi_i, t_i = run_independent()

    print(f"{'Mode':<26} | {'Computed pi':>12} | {'Abs error':>11} | {'Wall time (ms)':>14}")
    print("-" * 72)
    print(f"{'Shared PRNG + mutex':<26} | {pi_s:>12.8f} | {abs(pi_s - math.pi):>11.2e} | {t_s:>14.2f}")
    print(f"{'Independent PRNG/thread':<26} | {pi_i:>12.8f} | {abs(pi_i - math.pi):>11.2e} | {t_i:>14.2f}")
    print()
    print(f"math.pi = {math.pi:.8f}")
    print(f"Speedup from eliminating generator state locking: {t_s / t_i:.2f}x")


if __name__ == "__main__":
    main()