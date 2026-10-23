"""
Task 2 — Multi-Core Scaling vs. Amdahl's Law.
Student ID: 230103182
"""
import os

# Keep each NumPy call single-threaded: parallelism comes only from our workers
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103182
N = 20_000_000
P = 0.95          # parallelizable fraction for Amdahl's Law
REPEATS = 5       # take the best of several runs to reduce noise


def amdahl(p_workers: int, p_frac: float = P) -> float:
    """Theoretical max speedup: S = 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - p_frac) + p_frac / p_workers)


def chunk_sum(chunk: np.ndarray) -> float:
    # np.sum releases the GIL for float64, so threads run truly in parallel
    return float(np.sum(chunk))


def parallel_sum(data: np.ndarray, pool: ThreadPoolExecutor, workers: int) -> float:
    chunks = np.array_split(data, workers)   # views, no copying
    return sum(pool.map(chunk_sum, chunks))


def benchmark(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Returns (best wall time in ms, computed sum)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_sum(data, pool, workers)          # warm-up run
        best = float("inf")
        result = 0.0
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = parallel_sum(data, pool, workers)
            best = min(best, (time.perf_counter() - t0) * 1000)
    return best, result


def main():
    max_cores = os.cpu_count() or 1
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical cores (os.cpu_count()): {max_cores}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Array: {N:,} float64 | seed={STUDENT_ID} | P={P} | best of {REPEATS} runs")
    print()

    data = np.random.default_rng(STUDENT_ID).random(N, dtype=np.float64)

    worker_counts = [1, 2, 4, max_cores, max_cores * 2]
    seen = set()
    worker_counts = [w for w in worker_counts if not (w in seen or seen.add(w))]

    results = []
    for w in worker_counts:
        t_ms, s = benchmark(data, w)
        results.append((w, t_ms, s))

    t1 = results[0][1]
    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for w, t_ms, _ in results:
        print(f"{w:>6} | {t_ms:>14.2f} | {t1 / t_ms:>7.2f}x | {amdahl(w):>17.2f}x")
    print()

    sums = [s for _, _, s in results]
    print(f"Sum (1 worker): {sums[0]:.6f}")
    print(f"Max deviation across worker counts: {max(abs(s - sums[0]) for s in sums):.2e}")
    print(f"Amdahl limit as p -> infinity: {1 / (1 - P):.1f}x")


if __name__ == "__main__":
    main()