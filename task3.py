"""
Task 3 - Cache Line Contention (False Sharing)
Student ID: 230103182

Python threads cannot run in parallel because of the GIL, so this uses
separate PROCESSES that write into one block of shared memory. Each process
runs on its own core, which is what makes cache line contention possible.
"""
import os
import platform
import subprocess
import time
from multiprocessing import Event, Process
from multiprocessing.shared_memory import SharedMemory

STUDENT_ID = 230103182
ITERATIONS = 50_000_000
SLOT = 8      # bytes per counter (signed 64-bit int)
PAD = 16      # 16 x 8 bytes = 128-byte stride (>= 64-byte cache line)


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


# ---------- worker functions (run inside child processes) ----------

def worker_shared(shm_name: str, index: int, iters: int, start) -> None:
    """Increments shared_counts[index] directly in shared memory every time."""
    shm = SharedMemory(name=shm_name)
    counts = shm.buf.cast("q")
    start.wait()
    for _ in range(iters):
        counts[index] += 1
    del counts
    shm.close()


def worker_local(shm_name: str, index: int, iters: int, start) -> None:
    """Counts in a local variable, writes to shared memory only once."""
    shm = SharedMemory(name=shm_name)
    counts = shm.buf.cast("q")
    start.wait()
    local = 0
    for _ in range(iters):
        local += 1
    counts[index] = local
    del counts
    shm.close()


# ---------- benchmark driver ----------

def run(label: str, target, n: int, stride: int, iters: int) -> float:
    shm = SharedMemory(create=True, size=n * stride * SLOT)
    shm.buf[:] = bytes(len(shm.buf))
    start = Event()
    procs = [Process(target=target, args=(shm.name, i * stride, iters, start))
             for i in range(n)]
    for p in procs:
        p.start()
    time.sleep(1.0)               # let every process start up and reach start.wait()

    t0 = time.perf_counter()
    start.set()                   # all processes begin at the same moment
    for p in procs:
        p.join()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    counts = shm.buf.cast("q")
    ok = all(counts[i * stride] == iters for i in range(n))
    del counts
    shm.close()
    shm.unlink()
    print(f"  {label:<38} {elapsed_ms:>10.2f} ms   (counts correct: {ok})")
    return elapsed_ms


def main():
    logical = os.cpu_count() or 1
    n = physical_cores()

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical cores (os.cpu_count()): {logical}")
    print(f"Physical cores (workers used, N): {n}")
    print(f"Increments per worker: {ITERATIONS:,}")
    print()
    print("Running benchmarks...")

    t_unpadded = run("1. Unpadded shared (false sharing)", worker_shared, n, 1, ITERATIONS)
    t_padded = run(f"2. Padded shared ({PAD * SLOT}-byte stride)", worker_shared, n, PAD, ITERATIONS)
    t_local = run("3. Local variable, write once", worker_local, n, 1, ITERATIONS)

    print()
    header = f"{'Version':<36} | {'Wall Time (ms)':>14} | {'vs Unpadded':>11}"
    print(header)
    print("-" * len(header))
    print(f"{'Unpadded shared':<36} | {t_unpadded:>14.2f} | {1.0:>10.2f}x")
    print(f"{'Padded shared':<36} | {t_padded:>14.2f} | {t_unpadded / t_padded:>10.2f}x")
    print(f"{'Local variable':<36} | {t_local:>14.2f} | {t_unpadded / t_local:>10.2f}x")
    print()
    print(f"Execution ratio Unpadded / Padded: {t_unpadded / t_padded:.2f}x")
    print(f"Execution ratio Unpadded / Local : {t_unpadded / t_local:.2f}x")


if __name__ == "__main__":
    main()