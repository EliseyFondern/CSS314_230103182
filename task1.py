"""
Task 1 — Baseline serial execution.
Student ID: 230103182
"""
import os

# Force single-threaded math libraries BEFORE importing numpy
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import subprocess
import time

import numpy as np

STUDENT_ID = 230103182
N = 20_000_000


def get_cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL)
            if out.strip():
                return out.strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def main():
    # 1. System info
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {get_cpu_model()}")
    print(f"Logical cores (os.cpu_count()): {os.cpu_count()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print("-" * 50)

    # 2. Generate data with Student ID as seed
    rng = np.random.default_rng(STUDENT_ID)
    t0 = time.perf_counter()
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"Array size: {data.size:,} elements, dtype={data.dtype}")
    print(f"Generation time: {gen_ms:.2f} ms")

    # 3. Serial computation of sum and standard deviation
    t0 = time.perf_counter()
    total = np.sum(data)
    std = np.std(data)
    compute_ms = (time.perf_counter() - t0) * 1000

    # 4. Results and wall-clock time
    print(f"Sum: {total:.6f}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Execution time (sum + std, serial): {compute_ms:.2f} ms")


if __name__ == "__main__":
    main()