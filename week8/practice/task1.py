"""
TASK 1 — Row-Major vs. Column-Major Parallel Cache Traversal

Mode A (Row-Major):     each worker takes a block of ROWS and walks each row
                        left-to-right (contiguous memory, cache-friendly).
Mode B (Column-Major):  each worker takes a block of COLUMNS and walks each
                        column top-to-bottom (stride = 8000 * 8 bytes = 64 KB,
                        so every element touches a new cache line).
"""
import time
import os
import numpy as np
import numba
from numba import njit, prange
import psutil

# ---------------------------------------------------------------
STUDENT_ID = 230109009          # <-- PUT YOUR STUDENT ID HERE
N = 8000                       # 8,000 x 8,000 float64 = ~512 MB
RUNS = 5                       # timed repetitions (median reported)
OUTPUT_FILE = "task1_output.txt"
# ---------------------------------------------------------------


@njit(parallel=True)
def sum_row_major(A, nthreads):
    n_rows, n_cols = A.shape
    partial = np.zeros(nthreads)
    chunk = (n_rows + nthreads - 1) // nthreads
    for t in prange(nthreads):                 # one worker per row block
        start = t * chunk
        end = min(start + chunk, n_rows)
        s = 0.0
        for i in range(start, end):            # rows of this worker
            for j in range(n_cols):            # linear walk through columns
                s += A[i, j]
        partial[t] = s
    return partial.sum()


@njit(parallel=True)
def sum_col_major(A, nthreads):
    n_rows, n_cols = A.shape
    partial = np.zeros(nthreads)
    chunk = (n_cols + nthreads - 1) // nthreads
    for t in prange(nthreads):                 # one worker per column block
        start = t * chunk
        end = min(start + chunk, n_cols)
        s = 0.0
        for j in range(start, end):            # columns of this worker
            for i in range(n_rows):            # strided walk down the rows
                s += A[i, j]
        partial[t] = s
    return partial.sum()


def time_mode(fn, A, nthreads):
    times = []
    result = 0.0
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn(A, nthreads)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times)), result


def main():
    logical = psutil.cpu_count(logical=True) or os.cpu_count()
    physical = psutil.cpu_count(logical=False) or logical
    nthreads = max(1, min(physical, numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(nthreads)

    # 1. Allocate and fill the matrix using the Student ID as the seed
    rng = np.random.default_rng(STUDENT_ID)
    A = rng.random((N, N), dtype=np.float64)   # C-contiguous (row-major)

    # Warm-up: triggers JIT compilation so it is not counted in timing
    sum_row_major(A[:64, :64].copy(), nthreads)
    sum_col_major(A[:64, :64].copy(), nthreads)

    # 2-3. Parallel sums + wall-clock timing
    time_a, sum_a = time_mode(sum_row_major, A, nthreads)
    time_b, sum_b = time_mode(sum_col_major, A, nthreads)
    penalty = time_b / time_a

    lines = [
        "=" * 56,
        " TASK 1 - Spatial Cache Profiling (Row vs Column Major)",
        "=" * 56,
        f" Student ID              : {STUDENT_ID}",
        f" Logical core count      : {logical}",
        f" Physical core count     : {physical}",
        f" Worker threads used     : {nthreads}",
        f" Matrix                  : {N} x {N} float64 "
        f"({A.nbytes / 1024**2:.0f} MB)",
        f" Timed runs (median)     : {RUNS}",
        "-" * 56,
        f" Mode A (Row-Major)      : {time_a:10.2f} ms   sum = {sum_a:.6f}",
        f" Mode B (Column-Major)   : {time_b:10.2f} ms   sum = {sum_b:.6f}",
        f" Sums match              : {np.isclose(sum_a, sum_b)}",
        "-" * 56,
        f" Cache penalty factor    : {penalty:.2f}x  (Mode B / Mode A)",
        "=" * 56,
    ]
    report = "\n".join(lines)
    print(report)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()