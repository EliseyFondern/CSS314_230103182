"""
Task 3 - Prefix Scan (Step Efficiency vs. Work Efficiency)
Student ID: 230103182

Serial inclusive scan:  output[i] = output[i-1] + input[i]   (np.cumsum, one thread)
Parallel chunked scan (2-pass):
  Pass 1 (parallel): each thread scans its own chunk locally -> chunk totals
  Offset reduction (serial, tiny): exclusive scan of the chunk totals
  Pass 2 (parallel): each thread adds its chunk's offset to every element
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103182
N = 2_000_000
REPEATS = 10
WORKERS = os.cpu_count() or 1


def serial_scan(data: np.ndarray) -> np.ndarray:
    out = np.empty_like(data)
    np.cumsum(data, out=out)
    return out


def parallel_scan(data: np.ndarray, pool: ThreadPoolExecutor, bounds) -> np.ndarray:
    out = np.empty_like(data)

    # Pass 1: local inclusive scan of every chunk (NumPy releases the GIL -> parallel)
    def local_scan(b):
        lo, hi = b
        np.cumsum(data[lo:hi], out=out[lo:hi])
        return out[hi - 1]                       # chunk total
    totals = np.fromiter(pool.map(local_scan, bounds), dtype=data.dtype, count=len(bounds))

    # Offset reduction: exclusive scan of chunk totals (only `workers` numbers)
    offsets = np.concatenate(([0], np.cumsum(totals)[:-1]))

    # Pass 2: add the offset of all previous chunks to this chunk
    def apply_offset(args):
        (lo, hi), off = args
        if off:
            out[lo:hi] += off
    list(pool.map(apply_offset, zip(bounds, offsets)))
    return out


def best_ms(fn, *args) -> tuple[float, np.ndarray]:
    result = fn(*args)                           # warm-up
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {WORKERS}")

    data = np.random.default_rng(STUDENT_ID).integers(1, 11, size=N, dtype=np.int64)
    edges = np.linspace(0, N, WORKERS + 1, dtype=np.int64)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    print(f"Array: {N:,} int64 values in [1, 10] | chunks: {len(bounds)} | best of {REPEATS} runs")
    print(f"First 10 inputs: {data[:10].tolist()}")
    print()

    t_ser, serial_out = best_ms(serial_scan, data)
    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        t_par, parallel_out = best_ms(parallel_scan, data, pool, bounds)

    passed = bool(parallel_out[-1] == serial_out[-1])
    full_match = bool(np.array_equal(parallel_out, serial_out))
    status = "PASS" if passed else "FAIL"

    header = f"{'Serial Time (ms)':>16} | {'Parallel Time (ms)':>18} | {'Speedup':>8} | Verification Status"
    print(header)
    print("-" * len(header))
    print(f"{t_ser:>16.3f} | {t_par:>18.3f} | {t_ser / t_par:>7.2f}x | {status}")
    print()
    print(f"serial_output[-1]   = {int(serial_out[-1]):,}")
    print(f"parallel_output[-1] = {int(parallel_out[-1]):,}")
    print(f"Full array identical: {full_match}")
    print(f"Work: serial = {N:,} additions | parallel = ~{2 * N:,} additions "
          f"(2 passes) + {len(bounds)} offset adds")

    try:
        assert parallel_out[-1] == serial_out[-1], "parallel_output[-1] != serial_output[-1]"
    except AssertionError as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()