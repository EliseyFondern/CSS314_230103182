"""
Task 4 - Work Stealing & Scheduling Under Skew
Student ID: 230103182

1000 items, item i costs 0.0001 * (i % 20) seconds, 4 workers.
Strategy A: static equal blocks.  Strategy B: dynamic shared queue (chunk 1 and 5).
"""
import os
import platform
import threading
import time

STUDENT_ID = 230103182
N_ITEMS = 1000
WORKERS = 4
UNIT = 0.0001


def cost(i: int) -> float:
    return UNIT * (i % 20)


def simulate_work(seconds: float) -> None:
    """Precise 'sleep': time.sleep() on Windows rounds up to ~1 ms, which would
    distort costs of 0.1-1.9 ms. Waiting until a deadline is exact, and
    sleep(0) yields the GIL so the other threads keep working in parallel."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        time.sleep(0)


def run_static() -> dict:
    """Strategy A: worker k gets one contiguous block of N/WORKERS items."""
    block = N_ITEMS // WORKERS
    finish = [0.0] * WORKERS
    done = [0] * WORKERS
    planned = [0.0] * WORKERS

    def worker(k: int, t0: float) -> None:
        lo, hi = k * block, (k + 1) * block if k < WORKERS - 1 else N_ITEMS
        for i in range(lo, hi):
            simulate_work(cost(i))
            planned[k] += cost(i)
            done[k] += 1
        finish[k] = time.perf_counter() - t0

    return _launch(worker, finish, done, planned)


def run_dynamic(chunk: int) -> dict:
    """Strategy B: workers grab the next `chunk` items from a shared counter
    as soon as they are free (self-scheduling / work-sharing queue)."""
    lock = threading.Lock()
    next_index = [0]
    finish = [0.0] * WORKERS
    done = [0] * WORKERS
    planned = [0.0] * WORKERS

    def worker(k: int, t0: float) -> None:
        while True:
            with lock:
                lo = next_index[0]
                next_index[0] += chunk
            if lo >= N_ITEMS:
                break
            for i in range(lo, min(lo + chunk, N_ITEMS)):
                simulate_work(cost(i))
                planned[k] += cost(i)
                done[k] += 1
        finish[k] = time.perf_counter() - t0

    return _launch(worker, finish, done, planned)


def _launch(worker, finish, done, planned) -> dict:
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(k, t0)) for k in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return {"total": total * 1000, "finish": [f * 1000 for f in finish],
            "done": done, "planned": [p * 1000 for p in planned]}


def report(name: str, r: dict) -> None:
    delta = max(r["finish"]) - min(r["finish"])
    print(f"--- {name} ---")
    for k in range(WORKERS):
        print(f"  Worker {k}: items={r['done'][k]:>4} | assigned work={r['planned'][k]:>7.1f} ms"
              f" | finished at {r['finish'][k]:>7.2f} ms")
    print(f"  Total execution time      : {r['total']:.2f} ms")
    print(f"  First finish / Last finish: {min(r['finish']):.2f} ms / {max(r['finish']):.2f} ms")
    print(f"  Completion delta (max-min): {delta:.2f} ms")
    print()


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")
    total_work = sum(cost(i) for i in range(N_ITEMS)) * 1000
    print(f"Items: {N_ITEMS} | workers: {WORKERS} | cost(i) = {UNIT} * (i % 20) s")
    print(f"Total serial work: {total_work:.1f} ms | ideal parallel time: {total_work / WORKERS:.1f} ms")
    print()

    results = {
        "Strategy A: Static blocks (250 items each)": run_static(),
        "Strategy B: Dynamic queue, chunk = 1": run_dynamic(1),
        "Strategy B: Dynamic queue, chunk = 5": run_dynamic(5),
    }
    for name, r in results.items():
        report(name, r)

    header = f"{'Strategy':<30} | {'Total (ms)':>10} | {'Completion delta (ms)':>21}"
    print(header)
    print("-" * len(header))
    short = ["Static blocks", "Dynamic, chunk=1", "Dynamic, chunk=5"]
    for label, r in zip(short, results.values()):
        delta = max(r["finish"]) - min(r["finish"])
        print(f"{label:<30} | {r['total']:>10.2f} | {delta:>21.2f}")


if __name__ == "__main__":
    main()