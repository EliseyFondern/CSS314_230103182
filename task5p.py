"""
Task 5 - Thread Oversubscription Threshold
Student ID: 230103182

10,000 CPU-bound tasks: each one hashes a 1 KB string 500 times
(PBKDF2-HMAC-SHA256 with 500 iterations).

Why pbkdf2_hmac and not a loop of hashlib.sha256(): CPython only releases the
GIL inside sha256() for inputs >= 2 KB, so 1 KB hashes from many threads would
run one at a time. pbkdf2_hmac does all 500 rounds in C with the GIL released,
so the threads really compete for CPU cores.
"""
import hashlib
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103182
TASKS = 10_000
ROUNDS = 500
REPEATS = 3
TOLERANCE = 0.03            # ignore changes smaller than 3% (timing noise)

SALT = str(STUDENT_ID).encode()


def make_payloads() -> list[bytes]:
    """10,000 distinct 1 KB strings derived from the Student ID."""
    return [hashlib.shake_256(f"{STUDENT_ID}-{i}".encode()).digest(1024) for i in range(TASKS)]


def task(payload: bytes) -> bytes:
    # 500 rounds of HMAC-SHA256 keyed by the 1 KB string, all in C, GIL released
    return hashlib.pbkdf2_hmac("sha256", payload, SALT, ROUNDS)


def run_batch(payloads, pool_size: int) -> tuple[float, bytes]:
    best, digest = float("inf"), b""
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            results = list(pool.map(task, payloads, chunksize=1))
        best = min(best, time.perf_counter() - t0)
        digest = hashlib.sha256(b"".join(results)).digest()
    return best * 1000, digest


def main():
    cores = os.cpu_count() or 1
    pool_sizes = sorted({1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256})

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical cores (os.cpu_count()): {cores}")
    print(f"Batch: {TASKS:,} tasks x {ROUNDS} SHA-256 rounds over a 1 KB string | best of {REPEATS} runs")
    print(f"Pool sizes: {pool_sizes}")
    print()

    payloads = make_payloads()
    task(payloads[0])                                   # warm-up

    rows = []
    reference = None
    for size in pool_sizes:
        t_ms, digest = run_batch(payloads, size)
        reference = reference or digest
        assert digest == reference, "results differ between pool sizes"
        rows.append((size, t_ms))

    # Threshold: first pool size after the fastest one that is clearly slower
    t1 = rows[0][1]
    best_i = min(range(len(rows)), key=lambda i: rows[i][1])
    threshold_i = None
    for i in range(best_i + 1, len(rows)):
        if rows[i][1] > rows[best_i][1] * (1 + TOLERANCE):
            threshold_i = i
            break

    header = f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status"
    print(header)
    print("-" * 82)
    for i, (size, t_ms) in enumerate(rows):
        if i == best_i:
            status = "PEAK (fastest)"
        elif i == threshold_i:
            status = "<<< DEGRADATION STARTS: context switching overhead"
        elif threshold_i is not None and i > threshold_i:
            status = "oversubscribed (degraded)"
        elif i < best_i:
            status = "scaling up"
        else:
            status = "oversubscribed (plateau, within noise)"
        print(f"{size:>9} | {t_ms:>19.1f} | {t1 / t_ms:>18.2f}x | {status}")
    print()

    best_size, best_t = rows[best_i]
    print(f"Fastest pool size: {best_size} threads ({best_t:.1f} ms, {t1 / best_t:.2f}x vs 1 thread)")
    if threshold_i is not None:
        size, t_ms = rows[threshold_i]
        prev = rows[threshold_i - 1][0]
        print(f"Oversubscription threshold: execution time starts worsening at pool size {size} "
              f"({t_ms:.1f} ms, +{(t_ms / best_t - 1) * 100:.1f}% vs peak); "
              f"last size before it: {prev}")
    else:
        print(f"No degradation above {TOLERANCE:.0%} noise: extra threads beyond {best_size} "
              f"add no speed but cost little on this OS scheduler")
    print("All pool sizes produced identical hash results: True")


if __name__ == "__main__":
    main()