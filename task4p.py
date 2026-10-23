"""
Task 4 - Host-to-Device Transfer Penalty (or IPC Overhead)
Student ID: 230103182

GPU mode (CuPy available): N = [100k, 1M, 20M] float64,
    Time_H2D, Time_Kernel (vector multiply), Time_D2H, % of time on PCIe.
CPU mode (fallback, or --cpu): N = [100k, 1M, 10M] float64 passed to another process,
    Path A: multiprocessing Pipe (pickle serialization + copy)
    Path B: zero-copy multiprocessing.shared_memory (only the name is sent)

Usage:  python task4.py         (GPU if possible, otherwise CPU)
        python task4.py --cpu   (force CPU mode)
"""
import os
import platform
import sys
import time
from multiprocessing import Pipe, Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np

STUDENT_ID = 230103182
REPEATS = 5


def print_header():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")


# ======================================================================
# CPU MODE: IPC pickle vs zero-copy shared memory
# ======================================================================
def child(conn) -> None:
    """Receives arrays (Path A) or shared-memory handles (Path B) and acknowledges."""
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "pickle":
            arr = msg[1]                               # already unpickled = full copy
            conn.send(float(arr[-1]))
        elif kind == "shm":
            _, name, n = msg
            try:                                       # attach, no copy
                shm = SharedMemory(name=name, track=False)   # Python 3.13+
            except TypeError:
                shm = SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            conn.send(float(arr[-1]))
            del arr
            shm.close()


def run_cpu():
    sizes = [100_000, 1_000_000, 10_000_000]
    print_header()
    print("Mode: CPU inter-process transfer (parent -> child process)")
    print(f"Arrays: float64 | best of {REPEATS} round trips per size")
    print()

    parent_conn, child_conn = Pipe()
    proc = Process(target=child, args=(child_conn,))
    proc.start()
    rng = np.random.default_rng(STUDENT_ID)

    header = (f"{'N':>12} | {'Size':>8} | {'Path A: Pipe+pickle (ms)':>24} | "
              f"{'Path B: shared mem (ms)':>23} | {'A / B':>8}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        data = rng.random(n)
        expected = float(data[-1])

        # Path A: whole array is pickled, pushed through the pipe, unpickled in the child
        best_a = float("inf")
        for _ in range(REPEATS + 1):
            t0 = time.perf_counter()
            parent_conn.send(("pickle", data))
            got = parent_conn.recv()
            best_a = min(best_a, time.perf_counter() - t0)
            assert got == expected

        # Path B: array lives in shared memory; only its name crosses the process boundary
        shm = SharedMemory(create=True, size=data.nbytes)
        shared = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared[:] = data                                   # data placed once, outside timing
        best_b = float("inf")
        for _ in range(REPEATS + 1):
            t0 = time.perf_counter()
            parent_conn.send(("shm", shm.name, n))
            got = parent_conn.recv()
            best_b = min(best_b, time.perf_counter() - t0)
            assert got == expected
        del shared
        shm.close()
        shm.unlink()

        a_ms, b_ms = best_a * 1000, best_b * 1000
        size = f"{data.nbytes / 2**20:.1f} MB"
        print(f"{n:>12,} | {size:>8} | {a_ms:>24.3f} | {b_ms:>23.3f} | {a_ms / b_ms:>7.1f}x")

    parent_conn.send(("stop",))
    proc.join()
    print()
    print("Path A cost grows with N: the array is serialized, copied through the OS pipe, and rebuilt.")
    print("Path B cost is ~constant: only a short name is sent; both processes map the same RAM.")


# ======================================================================
# GPU MODE: H2D / kernel / D2H breakdown
# ======================================================================
def run_gpu(cp):
    sizes = [100_000, 1_000_000, 20_000_000]
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print_header()
    print(f"Mode: GPU (CUDA via CuPy {cp.__version__}) | GPU: {name}")
    print(f"Kernel: y = x * 2.5 (float64) | pageable host memory | median of {REPEATS} runs")
    print()

    scale = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * 2.5", "vec_mul")
    ev = [cp.cuda.Event() for _ in range(4)]
    rng = np.random.default_rng(STUDENT_ID)

    header = (f"{'N':>12} | {'H2D (ms)':>9} | {'Kernel (ms)':>11} | {'D2H (ms)':>9} | "
              f"{'Total (ms)':>10} | {'PCIe %':>7}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        host = rng.random(n)
        dev_out = cp.empty(n, dtype=cp.float64)
        h2d, ker, d2h = [], [], []
        for i in range(REPEATS + 1):
            ev[0].record()
            dev_in = cp.asarray(host)                      # Host -> Device
            ev[1].record()
            scale(dev_in, dev_out)                         # kernel
            ev[2].record()
            result = cp.asnumpy(dev_out)                   # Device -> Host
            ev[3].record()
            ev[3].synchronize()
            if i == 0:
                continue                                   # warm-up
            h2d.append(cp.cuda.get_elapsed_time(ev[0], ev[1]))
            ker.append(cp.cuda.get_elapsed_time(ev[1], ev[2]))
            d2h.append(cp.cuda.get_elapsed_time(ev[2], ev[3]))
        assert np.allclose(result, host * 2.5)
        t_h, t_k, t_d = (float(np.median(v)) for v in (h2d, ker, d2h))
        total = t_h + t_k + t_d
        pcie = (t_h + t_d) / total * 100
        print(f"{n:>12,} | {t_h:>9.3f} | {t_k:>11.3f} | {t_d:>9.3f} | {total:>10.3f} | {pcie:>6.1f}%")
    print()
    print("PCIe % = (H2D + D2H) / total: most of the pipeline is moving data, not computing.")


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