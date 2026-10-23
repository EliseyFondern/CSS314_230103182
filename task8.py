"""
Task 8 - Grid/Block Geometry (GPU) or Parallel Tile Reduction (CPU)
Student ID: 230103182

GPU mode (NVIDIA + CuPy available): vector dot product CUDA kernel,
    block sizes [32, 64, 128, 256, 512, 1024], latency per block geometry.
CPU mode (fallback, or run with --cpu): 1024x1024 matrix multiplication,
    non-tiled vs 64x64 vs 16x16 tiles, split across CPU threads.

Usage:  python task8.py         (GPU if possible, otherwise CPU)
        python task8.py --cpu   (force CPU mode)
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


def print_header():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")


# ======================================================================
# GPU MODE: dot product kernel with shared-memory tree reduction
# ======================================================================
DOT_KERNEL = r"""
extern "C" __global__
void dot(const double* __restrict__ x, const double* __restrict__ y,
         double* out, long long n)
{
    extern __shared__ double cache[];          // one slot per thread in the block
    unsigned int tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + tid;
    long long stride = (long long)gridDim.x * blockDim.x;

    double sum = 0.0;
    for (; i < n; i += stride)                 // grid-stride loop
        sum += x[i] * y[i];
    cache[tid] = sum;
    __syncthreads();

    // tree reduction inside the block: blockDim.x -> 1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, cache[0]);    // combine block results
}
"""


def run_gpu(cp):
    N = 10_000_000
    BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
    RUNS = 20

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print_header()
    print(f"Mode: GPU (CUDA via CuPy {cp.__version__})")
    print(f"GPU: {name} | SMs: {props['multiProcessorCount']} | "
          f"max threads/block: {props['maxThreadsPerBlock']}")
    print(f"Vector length: {N:,} float64 | median of {RUNS} runs per block size")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    x_host, y_host = rng.random(N), rng.random(N)
    x, y = cp.asarray(x_host), cp.asarray(y_host)
    out = cp.zeros(1, dtype=cp.float64)
    reference = float(np.dot(x_host, y_host))

    kernel = cp.RawKernel(DOT_KERNEL, "dot")
    start, stop = cp.cuda.Event(), cp.cuda.Event()

    header = f"{'Block size':>10} | {'Grid (blocks)':>13} | {'Latency (ms)':>12} | {'Bandwidth':>11} | {'Rel. error':>10}"
    print(header)
    print("-" * len(header))
    results = []
    for bs in BLOCK_SIZES:
        grid = (N + bs - 1) // bs
        args = (x, y, out, np.int64(N))
        shmem = bs * 8

        out.fill(0)
        kernel((grid,), (bs,), args, shared_mem=shmem)       # warm-up / compile
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(RUNS):
            out.fill(0)
            start.record()
            kernel((grid,), (bs,), args, shared_mem=shmem)
            stop.record()
            stop.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, stop))   # ms
        t = float(np.median(times))
        err = abs(float(out[0]) - reference) / abs(reference)
        gbps = (2 * N * 8) / (t / 1000) / 1e9
        results.append((bs, t))
        print(f"{bs:>10} | {grid:>13,} | {t:>12.4f} | {gbps:>6.1f} GB/s | {err:>10.2e}")

    best = min(results, key=lambda r: r[1])
    worst = max(results, key=lambda r: r[1])
    print()
    print(f"Fastest block geometry: {best[0]} threads/block ({best[1]:.4f} ms)")
    print(f"Slowest block geometry: {worst[0]} threads/block ({worst[1]:.4f} ms) "
          f"-> {worst[1] / best[1]:.2f}x slower")


# ======================================================================
# CPU MODE: tiled 1024x1024 matrix multiplication across threads
# ======================================================================
def matmul_non_tiled(A, B, C, rows):
    """Naive: each output row streams through the WHOLE of B (8 MB) -> poor cache reuse."""
    for i in rows:
        C[i, :] = (A[i, :, None] * B).sum(axis=0)


def make_tiled(tile):
    def matmul_tiled(A, B, C, rows):
        """Blocked: work on tile x tile sub-matrices that fit in L1/L2 cache."""
        n = A.shape[0]
        lo, hi = rows.start, rows.stop
        for i in range(lo, hi, tile):
            ie = min(i + tile, hi)
            for j in range(0, n, tile):
                acc = np.zeros((ie - i, min(tile, n - j)))
                for k in range(0, n, tile):
                    acc += A[i:ie, k:k + tile] @ B[k:k + tile, j:j + tile]
                C[i:ie, j:j + tile] = acc
    return matmul_tiled


def run_cpu():
    N = 1024
    workers = os.cpu_count() or 1
    print_header()
    print("Mode: CPU parallel (no CUDA/CuPy available or --cpu given)")
    print(f"Matrix: {N}x{N} float64 | threads: {workers} | rows split evenly across threads")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    A, B = rng.random((N, N)), rng.random((N, N))
    reference = A @ B

    # each thread gets a contiguous band of rows (multiple of 64 so tiles line up)
    band = max(64, ((N // workers) + 63) // 64 * 64)
    bands = [range(s, min(s + band, N)) for s in range(0, N, band)]

    variants = [
        ("Non-tiled (row x full B)", matmul_non_tiled),
        ("Tiled 64x64", make_tiled(64)),
        ("Tiled 16x16", make_tiled(16)),
    ]

    header = f"{'Variant':<26} | {'Runtime (ms)':>12} | {'vs non-tiled':>12} | {'Tile working set':>16} | Correct"
    print(header)
    print("-" * len(header))
    results = {}
    with ThreadPoolExecutor(max_workers=len(bands)) as pool:
        for name, fn in variants:
            C = np.zeros((N, N))
            t0 = time.perf_counter()
            list(pool.map(lambda r: fn(A, B, C, r), bands))
            t = (time.perf_counter() - t0) * 1000
            ok = np.allclose(C, reference)
            results[name] = t
            if "64" in name:
                ws = f"{3 * 64 * 64 * 8 // 1024} KB (L2)"
            elif "16" in name:
                ws = f"{3 * 16 * 16 * 8 // 1024} KB (L1)"
            else:
                ws = f"{N * N * 8 // 2**20} MB (> L2)"
            base = results["Non-tiled (row x full B)"]
            print(f"{name:<26} | {t:>12.1f} | {base / t:>11.2f}x | {ws:>16} | {ok}")

    best = min(results, key=results.get)
    print()
    print(f"Fastest: {best} ({results[best]:.1f} ms)")
    print("Non-tiled re-reads all of B (8 MB, larger than L2) for every row -> constant cache misses.")
    print("64x64 tiles (96 KB) stay in L2 and are reused many times -> far fewer misses to RAM.")
    print("16x16 tiles (6 KB) fit in L1, but are so small that per-tile overhead dominates.")


if __name__ == "__main__":
    if "--cpu" in sys.argv:
        run_cpu()
    else:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            run_gpu(cp)
        except Exception as e:
            print(f"[GPU unavailable: {type(e).__name__}: {e}] -> falling back to CPU mode\n")
            run_cpu()