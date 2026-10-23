"""
Task 6 - Synchronization Cost & Data Loss
Student ID: 230103182

Why processes instead of threads: with CPython's GIL, threads switch only at
specific bytecode points, so `counter += 1` from 4 threads almost never loses
updates and the race cannot be shown. Four PROCESSES sharing one counter in
shared memory run truly in parallel on different cores, exactly like threads
in C/Java, so the read-modify-write race is real.
"""
import os
import platform
import time
from multiprocessing import Event, Lock, Process
from multiprocessing.shared_memory import SharedMemory

STUDENT_ID = 230103182
WORKERS = 4
INCREMENTS = 2_000_000
EXPECTED = WORKERS * INCREMENTS


def worker_unsafe(shm_name: str, iters: int, start) -> None:
    """counter += 1 with NO synchronization: read, add, write are separate steps."""
    shm = SharedMemory(name=shm_name)
    counter = shm.buf.cast("q")
    start.wait()
    for _ in range(iters):
        counter[0] += 1          # read -> +1 -> write : another core can interleave
    del counter
    shm.close()


def worker_locked(shm_name: str, iters: int, start, lock) -> None:
    """Same increment, protected by a mutex (critical section)."""
    shm = SharedMemory(name=shm_name)
    counter = shm.buf.cast("q")
    start.wait()
    for _ in range(iters):
        with lock:
            counter[0] += 1
    del counter
    shm.close()


def run(target, extra_args=()) -> tuple[int, float]:
    shm = SharedMemory(create=True, size=8)
    shm.buf[:] = bytes(8)                      # counter = 0
    start = Event()
    procs = [Process(target=target, args=(shm.name, INCREMENTS, start, *extra_args))
             for _ in range(WORKERS)]
    for p in procs:
        p.start()
    time.sleep(1.0)                            # exclude process start-up from timing

    t0 = time.perf_counter()
    start.set()
    for p in procs:
        p.join()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    counter = shm.buf.cast("q")
    value = counter[0]
    del counter
    shm.close()
    shm.unlink()
    return value, elapsed_ms


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")
    print(f"Workers: {WORKERS} parallel processes | increments each: {INCREMENTS:,}")
    print()

    # --- 1. No synchronization ---
    actual, t_unsafe = run(worker_unsafe)
    lost = EXPECTED - actual
    print("=== Version 1: NO synchronization (race condition) ===")
    print(f"  Expected final value : {EXPECTED:,}")
    print(f"  Actual final value   : {actual:,}")
    print(f"  Lost updates         : {lost:,} ({lost / EXPECTED * 100:.2f}%)")
    print(f"  Wall-clock time      : {t_unsafe:.2f} ms")
    print()

    # --- 2. Mutex / Lock ---
    lock = Lock()
    actual_l, t_locked = run(worker_locked, (lock,))
    lost_l = EXPECTED - actual_l
    print("=== Version 2: Mutex / Lock (correct) ===")
    print(f"  Expected final value : {EXPECTED:,}")
    print(f"  Actual final value   : {actual_l:,}")
    print(f"  Lost updates         : {lost_l:,} ({lost_l / EXPECTED * 100:.2f}%)")
    print(f"  Wall-clock time      : {t_locked:.2f} ms")
    print()

    # --- 3. Cost of correctness ---
    print("=== Synchronization penalty ===")
    print(f"  Unsynchronized : {t_unsafe:>10.2f} ms  (WRONG result)")
    print(f"  With lock      : {t_locked:>10.2f} ms  (correct result)")
    print(f"  Time penalty   : +{t_locked - t_unsafe:.2f} ms ({t_locked / t_unsafe:.2f}x slower)")
    print(f"  Cost per locked increment: {(t_locked - t_unsafe) * 1e6 / EXPECTED:.1f} ns extra")


if __name__ == "__main__":
    main()