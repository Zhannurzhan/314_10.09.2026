"""
TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks
Kernel A (memory-bound):  B[i] = A[i]
Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])
"""

import os

# NumPy's own libraries must not add threads; only OUR threads count.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import numpy as np

# ===================== CHANGE THIS =====================
STUDENT_ID = "230109009"  
# =======================================================

N = 30_000_000
MAX_CORES = os.cpu_count() or 1
CORE_COUNTS = sorted({1, 2, MAX_CORES})
REPEATS = 5                 # median of 5 timed runs (after 1 warm-up)
OUTPUT_FILE = "task7_output.txt"

lines = []


def log(text=""):
    """Print to the console AND remember the line for task7_output.txt."""
    print(text)
    lines.append(text)


# ---------------- The two kernels (work on one slice) ----------------
def kernel_copy(a, b, tmp, lo, hi):
    """Kernel A: B[i] = A[i]   -> 1 read + 1 write per element, NO math."""
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_trig(a, b, tmp, lo, hi):
    """Kernel B: B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])   -> lots of math."""
    x, out, t = a[lo:hi], b[lo:hi], tmp[lo:hi]
    np.sin(x, out=out)
    np.cos(x, out=t)
    np.multiply(out, t, out=out)
    np.tan(x, out=t)
    np.add(out, t, out=out)


def run(kernel, a, b, tmp, pool, workers):
    """Split the array into equal slices, one thread per slice."""
    bounds = np.linspace(0, N, workers + 1, dtype=np.int64)
    futures = [pool.submit(kernel, a, b, tmp, bounds[i], bounds[i + 1])
               for i in range(workers)]
    for f in futures:
        f.result()


def benchmark(kernel, a, b, tmp):
    """Return {cores: median time in ms}."""
    times = {}
    for cores in CORE_COUNTS:
        with ThreadPoolExecutor(max_workers=cores) as pool:
            run(kernel, a, b, tmp, pool, cores)              # warm-up (not timed)
            samples = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                run(kernel, a, b, tmp, pool, cores)
                samples.append((time.perf_counter() - start) * 1000)
        times[cores] = median(samples)
    return times


def print_table(title, times, bytes_moved=None):
    t1 = times[1]
    log(title)
    header = f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}"
    if bytes_moved:
        header += f" | {'Bandwidth':>11}"
    log(header)
    log("-" * 66)
    stats = {}
    for cores, t in times.items():
        speedup = t1 / t
        eff = speedup / cores
        stats[cores] = (speedup, eff)
        row = f"{cores:>6} | {t:>10.2f} | {speedup:>7.2f}x | {eff * 100:>9.1f}%"
        if bytes_moved:
            row += f" | {bytes_moved / (t / 1000) / 1e9:>7.2f} GB/s"
        log(row)
    log("-" * 66)
    return stats


def main():
    digits = "".join(ch for ch in STUDENT_ID if ch.isdigit())
    rng = np.random.default_rng(int(digits) if digits else 0)

    log("=" * 66)
    log("TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks")
    log("=" * 66)
    log(f"Student ID       : {STUDENT_ID}")
    log(f"OS               : {platform.platform()}")
    log(f"Python / NumPy   : {platform.python_version()} / {np.__version__}")
    log(f"Logical cores    : {MAX_CORES}  -> testing cores {CORE_COUNTS}")
    log(f"Array size       : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array)")
    log(f"Timing           : median of {REPEATS} runs after 1 warm-up")
    log("Efficiency       : Speedup / Cores   (100% = perfect scaling)")
    log("=" * 66)

    a = rng.uniform(0.0, 1.0, N)
    b = np.zeros(N)            # zeros() + warm-up makes sure memory pages are
    tmp = np.zeros(N)          # really allocated before we start timing

    # Kernel A moves 8 bytes in + 8 bytes out per element
    copy_stats = print_table("KERNEL A - Memory-bound:  B[i] = A[i]",
                             benchmark(kernel_copy, a, b, tmp), bytes_moved=16 * N)
    if not np.array_equal(a, b):
        raise RuntimeError("Copy kernel gave a wrong result")
    log()

    trig_stats = print_table("KERNEL B - Compute-bound: B[i] = sin(A[i])*cos(A[i]) + tan(A[i])",
                             benchmark(kernel_trig, a, b, tmp))
    check = np.sin(a[:1000]) * np.cos(a[:1000]) + np.tan(a[:1000])
    if not np.allclose(b[:1000], check):
        raise RuntimeError("Compute kernel gave a wrong result")

    # ---------------- Verdict ----------------
    m = MAX_CORES
    sp_a, eff_a = copy_stats[m]
    sp_b, eff_b = trig_stats[m]
    log("=" * 66)
    log(f"COMPARISON AT {m} CORES")
    log(f"  Kernel A (copy)    : speedup {sp_a:5.2f}x, efficiency {eff_a * 100:5.1f}%")
    log(f"  Kernel B (sin/cos) : speedup {sp_b:5.2f}x, efficiency {eff_b * 100:5.1f}%")
    log("-" * 66)
    if eff_a < eff_b:
        log("VERDICT: Kernel A (memory-bound copy) STALLED in scaling due to")
        log("         memory channel saturation. It does almost no math, so adding")
        log("         cores cannot help once the RAM bus is full: every core is")
        log("         waiting for data, not computing.")
        log(f"         Kernel B scales better ({eff_b * 100:.1f}% vs {eff_a * 100:.1f}% efficiency)")
        log("         because each element needs heavy math, so cores stay busy")
        log("         computing instead of waiting for RAM.")
    else:
        log("VERDICT: Unexpected result - Kernel B scaled worse than Kernel A on")
        log("         this run. Close other programs and run again.")
    log("=" * 66)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(Output also saved to {OUTPUT_FILE})")


if __name__ == "__main__":
    main()