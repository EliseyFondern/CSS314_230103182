"""
Task 1 - Row-Major vs. Column-Major Parallel Cache Traversal
Student ID: 230103182

NumPy arrays are row-major (C order): M[r, 0], M[r, 1], ... are adjacent in memory.
Mode A walks rows (contiguous, cache-friendly).
Mode B walks columns (each next element is 8000 * 8 = 64,000 bytes away).
"""
import os

# Each NumPy call stays single-threaded; parallelism comes only from our workers
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103182
SIZE = 8_000
REPEATS = 3


def physical_cores() -> int:
    logical = os.cpu_count() or 1
    try:
        system = platform.system()
        if system == "Windows":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]
        elif system == "Darwin":
            cmd = ["sysctl", "-n", "hw.physicalcpu"]
        else:
            cmd = ["sh", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        value = int(out.splitlines()[-1])
        if value > 0:
            return value
    except Exception:
        pass
    return max(1, logical // 2)


def sum_rows(M: np.ndarray, lo: int, hi: int) -> float:
    """Mode A: worker owns rows lo..hi, reads each row left->right (contiguous)."""
    total = 0.0
    for r in range(lo, hi):
        total += float(M[r, :].sum())
    return total


def sum_cols(M: np.ndarray, lo: int, hi: int) -> float:
    """Mode B: worker owns columns lo..hi, reads each column top->bottom (strided)."""
    total = 0.0
    for c in range(lo, hi):
        total += float(M[:, c].sum())
    return total


def run_mode(fn, M, pool, workers) -> tuple[float, float]:
    edges = np.linspace(0, SIZE, workers + 1, dtype=np.int64)
    slices = list(zip(edges[:-1], edges[1:]))
    best, result = float("inf"), 0.0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        # NumPy reductions release the GIL, so the threads run in parallel
        result = sum(pool.map(lambda s: fn(M, s[0], s[1]), slices))
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main():
    logical = os.cpu_count() or 1
    workers = physical_cores()

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {logical}")
    print(f"Physical cores (workers used): {workers}")
    print(f"Matrix: {SIZE:,} x {SIZE:,} float64 ({SIZE * SIZE * 8 / 2**20:.0f} MB), "
          f"row-major (C order) | best of {REPEATS} runs")
    print()

    M = np.random.default_rng(STUDENT_ID).random((SIZE, SIZE), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        run_mode(sum_rows, M, pool, workers)            # warm-up (page in memory)
        t_a, sum_a = run_mode(sum_rows, M, pool, workers)
        t_b, sum_b = run_mode(sum_cols, M, pool, workers)

    print(f"{'Mode':<44} | {'Runtime (ms)':>12} | {'Sum':>18}")
    print("-" * 82)
    print(f"{'A: Row-Major (rows sliced, contiguous)':<44} | {t_a:>12.2f} | {sum_a:>18.4f}")
    print(f"{'B: Column-Major (columns sliced, strided)':<44} | {t_b:>12.2f} | {sum_b:>18.4f}")
    print()
    print(f"Mode A runtime: {t_a:.2f} ms")
    print(f"Mode B runtime: {t_b:.2f} ms")
    print(f"Cache penalty factor (B / A): {t_b / t_a:.2f}x")
    print(f"Sums match: {abs(sum_a - sum_b) / abs(sum_a) < 1e-9}")
    print(f"Stride in Mode B: {SIZE * 8:,} bytes between consecutive elements "
          f"(vs 8 bytes in Mode A; cache line = 64 bytes)")


if __name__ == "__main__":
    main()