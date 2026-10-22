"""
TASK 5 - The Overhead Crossover Point
Benchmark y[i] = a * x[i] + b   serial vs. multi-threaded.
"""

import os

# NumPy's own libraries must not add threads; only OUR threads count.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ===================== CHANGE THIS =====================
STUDENT_ID = "230109009"   
# =======================================================

SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A, B = 2.5, 1.5
WORKERS = os.cpu_count() or 1
REPEATS = 7              # timing rounds per measurement; the fastest is kept
MIN_ROUND_TIME = 0.02    # each round repeats the operation for at least 20 ms
OUTPUT_FILE = "task5_output.txt"

lines = []


def log(text=""):
    """Print to the console AND remember the line for task5_output.txt."""
    print(text)
    lines.append(text)


# ---------------- The vector operation ----------------
def axpb(x, y, lo, hi):
    """y[lo:hi] = A * x[lo:hi] + B, computed in place (no extra arrays)."""
    np.multiply(x[lo:hi], A, out=y[lo:hi])
    np.add(y[lo:hi], B, out=y[lo:hi])


def serial(x, y, _pool, _bounds):
    axpb(x, y, 0, len(x))


def parallel(x, y, pool, bounds):
    """Split the array into WORKERS equal slices, one thread per slice."""
    futures = [pool.submit(axpb, x, y, bounds[i], bounds[i + 1])
               for i in range(WORKERS)]
    for f in futures:
        f.result()


def time_per_call(fn, x, y, pool, bounds):
    """Return the time of ONE call in microseconds.
    Small N runs in microseconds, so we repeat it many times per round
    and keep the fastest round (least disturbed by the OS)."""
    loops = 1
    while True:                                  # find how many loops fill ~20 ms
        start = time.perf_counter()
        for _ in range(loops):
            fn(x, y, pool, bounds)
        if time.perf_counter() - start >= MIN_ROUND_TIME:
            break
        loops *= 2
    best = math.inf
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(loops):
            fn(x, y, pool, bounds)
        best = min(best, (time.perf_counter() - start) / loops)
    return best * 1e6


def measure(n, pool, rng):
    """Time serial and parallel for size n. Returns (serial_us, parallel_us)."""
    x = rng.random(n)
    y_ser = np.empty(n)
    y_par = np.empty(n)
    bounds = np.linspace(0, n, WORKERS + 1, dtype=np.int64)

    t_ser = time_per_call(serial, x, y_ser, pool, bounds)
    t_par = time_per_call(parallel, x, y_par, pool, bounds)

    if not np.allclose(y_ser, y_par):            # both must give the same answer
        raise RuntimeError(f"Parallel result differs from serial at N={n}")
    return t_ser, t_par


def fmt_time(us):
    return f"{us:,.1f} us" if us < 10_000 else f"{us / 1000:,.2f} ms"


def main():
    digits = "".join(ch for ch in STUDENT_ID if ch.isdigit())
    rng = np.random.default_rng(int(digits) if digits else 0)

    log("=" * 74)
    log("TASK 5 - The Overhead Crossover Point:  y[i] = a * x[i] + b")
    log("=" * 74)
    log(f"Student ID       : {STUDENT_ID}")
    log(f"OS               : {platform.platform()}")
    log(f"Python / NumPy   : {platform.python_version()} / {np.__version__}")
    log(f"Logical cores    : {WORKERS}  -> parallel version uses {WORKERS} threads")
    log(f"a, b             : {A}, {B}")
    log(f"Timing           : best of {REPEATS} rounds, each round >= {MIN_ROUND_TIME * 1000:.0f} ms")
    log("=" * 74)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # warm-up: start all pool threads before measuring
        measure(10_000, pool, rng)

        # ---------- Part 1: the required sizes ----------
        log(f"{'N':>12} | {'Serial':>12} | {'Parallel':>12} | {'Speedup':>8} | Result")
        log("-" * 74)
        results = []
        for n in SIZES:
            t_ser, t_par = measure(n, pool, rng)
            speedup = t_ser / t_par
            faster = t_par < t_ser
            results.append((n, t_ser, t_par, faster))
            log(f"{n:>12,} | {fmt_time(t_ser):>12} | {fmt_time(t_par):>12} | "
                f"{speedup:>7.2f}x | {'PARALLEL FASTER' if faster else 'parallel slower'}")
        log("-" * 74)

        # ---------- Part 2: find the crossover range ----------
        # Crossover = smallest tested N where parallel is faster
        # AND stays faster for every larger N.
        first_fast = None
        for idx in range(len(results)):
            if all(r[3] for r in results[idx:]):
                first_fast = idx
                break

        if first_fast is None:
            log("RESULT: Multi-threading was NEVER faster than serial for these sizes.")
            log(f"        The crossover is above N = {SIZES[-1]:,}.")
        elif first_fast == 0:
            log("RESULT: Multi-threading was faster for ALL sizes tested.")
            log(f"        The crossover is below N = {SIZES[0]:,}.")
        else:
            lo_n, hi_n = SIZES[first_fast - 1], SIZES[first_fast]
            log(f"RESULT: Multi-threading shifts from SLOWER to FASTER than serial")
            log(f"        between N = {lo_n:,} (slower) and N = {hi_n:,} (faster).")
            log("-" * 74)

            # ---------- Part 3: narrow it down (bisection on a log scale) ----------
            log(f"Refining the tipping point between {lo_n:,} and {hi_n:,}:")
            log(f"{'N':>12} | {'Serial':>12} | {'Parallel':>12} | {'Speedup':>8}")
            while hi_n / lo_n > 1.05:
                mid = int(round(math.sqrt(lo_n * hi_n)))
                if mid in (lo_n, hi_n):
                    break
                t_ser, t_par = measure(mid, pool, rng)
                log(f"{mid:>12,} | {fmt_time(t_ser):>12} | {fmt_time(t_par):>12} | "
                    f"{t_ser / t_par:>7.2f}x")
                if t_par < t_ser:
                    hi_n = mid
                else:
                    lo_n = mid
            log("-" * 74)
            log(f"TIPPING POINT: parallel becomes faster at about N = {hi_n:,}")
            log(f"               (slower at N = {lo_n:,}, faster at N = {hi_n:,})")

    log("=" * 74)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(Output also saved to {OUTPUT_FILE})")


if __name__ == "__main__":
    main()