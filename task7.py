"""
Task 7 - Arithmetic Intensity vs. Memory Bottlenecks
Student ID: 230103182

Kernel A (memory-bound):  B[i] = A[i]
Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])
Both run with 1, 2 and max cores; efficiency = speedup / cores.
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
N = 30_000_000
REPEATS = 3


def kernel_copy(a, b, t):
    """Memory-bound: 1 read + 1 write per element, zero arithmetic."""
    np.copyto(b, a)


def kernel_trig(a, b, t):
    """Compute-bound: 3 transcendental functions per element, in place."""
    np.sin(a, out=b)
    np.cos(a, out=t)
    np.multiply(b, t, out=b)
    np.tan(a, out=t)
    np.add(b, t, out=b)


def run(kernel, A, B, T, workers: int) -> float:
    """Best wall time (ms) of `kernel` split across `workers` threads."""
    edges = np.linspace(0, len(A), workers + 1, dtype=np.int64)
    parts = [(A[lo:hi], B[lo:hi], T[lo:hi]) for lo, hi in zip(edges[:-1], edges[1:])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda p: kernel(*p), parts))          # warm-up
        best = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            # NumPy ufuncs release the GIL -> true parallel execution
            list(pool.map(lambda p: kernel(*p), parts))
            best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    max_cores = os.cpu_count() or 1
    core_counts = sorted({1, 2, max_cores})

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {max_cores}")
    print(f"Array: {N:,} float64 ({N * 8 / 1e6:.0f} MB per array) | best of {REPEATS} runs")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    A = rng.random(N)
    B = np.empty_like(A)
    T = np.empty_like(A)

    kernels = {
        "A: Copy  B[i] = A[i]  (memory-bound)": kernel_copy,
        "B: sin*cos+tan        (compute-bound)": kernel_trig,
    }

    header = f"{'Cores':>5} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'Bandwidth':>11}"
    efficiency_at_max = {}
    for name, kernel in kernels.items():
        print(f"--- Kernel {name} ---")
        print(header)
        print("-" * len(header))
        t1 = None
        for c in core_counts:
            t = run(kernel, A, B, T, c)
            t1 = t1 or t
            speedup = t1 / t
            eff = speedup / c
            # bytes moved: copy = 1 read + 1 write; trig reads A 3x, touches B/T several times
            gbps = (2 * N * 8) / (t / 1000) / 1e9 if kernel is kernel_copy else None
            bw = f"{gbps:>8.1f} GB/s" if gbps else f"{'-':>11}"
            print(f"{c:>5} | {t:>10.2f} | {speedup:>7.2f}x | {eff * 100:>9.1f}% | {bw}")
            efficiency_at_max[name] = (speedup, eff)
        print()

    # Correctness check for the trig kernel
    sample = A[:1000]
    run(kernel_trig, A, B, T, max_cores)
    assert np.allclose(B[:1000], np.sin(sample) * np.cos(sample) + np.tan(sample))

    (name_a, (s_a, e_a)), (name_b, (s_b, e_b)) = efficiency_at_max.items()
    stalled, scaled = (name_a, name_b) if e_a < e_b else (name_b, name_a)
    print("=== Conclusion ===")
    print(f"At {max_cores} cores: Kernel A speedup {s_a:.2f}x (efficiency {e_a * 100:.1f}%), "
          f"Kernel B speedup {s_b:.2f}x (efficiency {e_b * 100:.1f}%)")
    print(f"STALLED due to memory channel saturation: Kernel {' '.join(stalled.split())}")
    print(f"  -> almost no arithmetic per byte; extra cores just wait on the RAM bus.")
    print(f"SCALED better: Kernel {' '.join(scaled.split())}")
    print(f"  -> high arithmetic intensity; each core keeps busy computing, so more cores help.")


if __name__ == "__main__":
    main()