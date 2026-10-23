"""
Task 5 - The Overhead Crossover Point
Student ID: 230103182

y[i] = a * x[i] + b, serial vs. thread pool using all logical cores.
Finds the range of N where the parallel version goes from SLOWER to FASTER.
"""
import os

# Each NumPy call stays single-threaded; parallelism comes only from our pool
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103182
A, B = 2.5, 1.5
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORKERS = os.cpu_count() or 1


def axpb(x: np.ndarray, y: np.ndarray) -> None:
    """y = a*x + b computed in place (no temporary arrays)."""
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def serial(x, y, _pool, _bounds):
    axpb(x, y)


def parallel(x, y, pool, bounds):
    # NumPy ufuncs release the GIL, so threads run truly in parallel.
    # Slices are views: no data is copied.
    futures = [pool.submit(axpb, x[lo:hi], y[lo:hi]) for lo, hi in bounds]
    for f in futures:
        f.result()


def repeats_for(n: int) -> int:
    # Small arrays are very fast and noisy -> measure many times
    return max(5, min(2000, 20_000_000 // max(n, 1)))


def best_time_ms(fn, x, y, pool, bounds) -> float:
    reps = repeats_for(len(x))
    fn(x, y, pool, bounds)                      # warm-up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x, y, pool, bounds)
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def measure(n: int, pool, rng) -> tuple[float, float]:
    x = rng.random(n)
    y = np.empty_like(x)
    workers = min(WORKERS, n)
    edges = np.linspace(0, n, workers + 1, dtype=np.int64)
    bounds = list(zip(edges[:-1], edges[1:]))

    t_ser = best_time_ms(serial, x, y, pool, bounds)
    expected = A * x + B
    t_par = best_time_ms(parallel, x, y, pool, bounds)
    assert np.allclose(y, expected), "parallel result mismatch"
    return t_ser, t_par


def print_row(n, t_ser, t_par):
    speedup = t_ser / t_par
    verdict = "FASTER" if t_par < t_ser else "SLOWER"
    print(f"{n:>12,} | {t_ser:>11.4f} | {t_par:>13.4f} | {speedup:>7.2f}x | parallel {verdict}")


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores / pool workers: {WORKERS}")
    print(f"Operation: y[i] = {A} * x[i] + {B} (float64) | best of many runs")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    header = f"{'N':>12} | {'Serial (ms)':>11} | {'Parallel (ms)':>13} | {'Speedup':>8} | Result"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        print(header)
        print("-" * len(header))
        results = []
        for n in SIZES:
            t_ser, t_par = measure(n, pool, rng)
            results.append((n, t_ser, t_par))
            print_row(n, t_ser, t_par)
        print()

        # Find where parallel switches from slower to faster
        crossover = None
        for (n0, s0, p0), (n1, s1, p1) in zip(results, results[1:]):
            if p0 >= s0 and p1 < s1:
                crossover = (n0, n1)
                break

        if crossover is None:
            if all(p < s for _, s, p in results):
                print("Parallel was FASTER for every tested N (crossover is below 100).")
            else:
                print("Parallel never became FASTER than serial in the tested range.")
            return

        lo, hi = crossover
        print(f"CROSSOVER (coarse): parallel becomes faster between N = {lo:,} and N = {hi:,}")
        print()

        # Refine with 8 log-spaced sizes inside the coarse range
        print("Refining the crossover inside that range:")
        print(header)
        print("-" * len(header))
        refine = sorted({int(v) for v in np.geomspace(lo, hi, 10)[1:-1]})
        last_slower, first_faster = lo, hi
        for n in refine:
            t_ser, t_par = measure(n, pool, rng)
            print_row(n, t_ser, t_par)
            if t_par >= t_ser:
                last_slower = n
            elif first_faster == hi:
                first_faster = n
        if first_faster < last_slower:          # noisy region: report the full overlap
            first_faster, last_slower = last_slower, first_faster
        print()
        print(f"CROSSOVER (refined): parallel shifts from SLOWER to FASTER between "
              f"N = {last_slower:,} and N = {first_faster:,} on this machine")


if __name__ == "__main__":
    main()