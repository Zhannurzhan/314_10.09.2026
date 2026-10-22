"""
TASK 2 - Multi-Core Scaling vs. Amdahl's Law
"""

import os

# Stop NumPy's own internal libraries from using extra threads,
# so the ONLY parallelism comes from our worker threads.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import numpy as np

# ===================== CHANGE THIS =====================
STUDENT_ID = "230109009"   # <-- your student ID
# =======================================================

N = 20_000_000
P = 0.95          # parallel fraction for Amdahl's Law
REPEATS = 5       # each test runs 5 times; we keep the median time
OUTPUT_FILE = "task2_output.txt"

lines = []


def log(text=""):
    """Print to the console AND remember the line for task2_output.txt."""
    print(text)
    lines.append(text)


def chunk_sum(data, start, end):
    """Sum one slice of the array. NumPy releases the GIL here,
    so several threads really run at the same time."""
    return np.sum(data[start:end])


def parallel_sum(data, pool, workers):
    """Split the array into equal parts, sum each part in its own thread,
    then add the partial sums together."""
    bounds = np.linspace(0, len(data), workers + 1, dtype=np.int64)
    futures = [pool.submit(chunk_sum, data, bounds[i], bounds[i + 1])
               for i in range(workers)]
    return sum(f.result() for f in futures)


def amdahl(p_cores, parallel_fraction=P):
    """Amdahl's Law: S(p) = 1 / ((1 - P) + P / p)"""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_cores)


def main():
    digits = "".join(ch for ch in STUDENT_ID if ch.isdigit())
    seed = int(digits) if digits else 0
    max_cores = os.cpu_count() or 1

    # Worker counts: [1, 2, 4, max_cores, max_cores*2], without duplicates
    worker_counts = []
    for w in (1, 2, 4, max_cores, max_cores * 2):
        if w not in worker_counts:
            worker_counts.append(w)

    log("=" * 62)
    log("TASK 2 - Multi-Core Scaling vs. Amdahl's Law")
    log("=" * 62)
    log(f"Student ID       : {STUDENT_ID}")
    log(f"OS               : {platform.platform()}")
    log(f"Logical cores    : {max_cores}")
    log(f"Array size       : {N:,} float64 elements (seed = {seed})")
    log(f"Parallel model   : threads (ThreadPoolExecutor), NumPy releases the GIL")
    log(f"Amdahl P         : {P}")
    log(f"Repeats per test : {REPEATS} (median time reported)")
    log("-" * 62)

    data = np.random.default_rng(seed).random(N, dtype=np.float64)
    reference = np.sum(data)   # serial answer, used to check correctness

    results = []
    for workers in worker_counts:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parallel_sum(data, pool, workers)          # warm-up run (not timed)
            times = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                total = parallel_sum(data, pool, workers)
                times.append((time.perf_counter() - start) * 1000)
        if not np.isclose(total, reference, rtol=1e-9):
            raise RuntimeError(f"Wrong sum with {workers} workers: {total}")
        results.append((workers, median(times), total))

    t1 = results[0][1]   # T(1)

    log(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}")
    log("-" * 62)
    for workers, t, _ in results:
        log(f"{workers:>6} | {t:>14.2f} | {t1 / t:>7.2f}x | {amdahl(workers):>17.2f}x")
    log("-" * 62)
    log(f"Sum (check)      : {results[-1][2]:.6f}  (matches serial: {reference:.6f})")
    log(f"Amdahl limit     : max speedup with infinite cores = {1 / (1 - P):.0f}x")
    log("=" * 62)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(Output also saved to {OUTPUT_FILE})")


if __name__ == "__main__":
    main()