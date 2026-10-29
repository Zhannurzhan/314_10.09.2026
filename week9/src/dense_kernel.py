#!/usr/bin/env python3
"""
dense_kernel.py -- Task 1.2: Loop Interchange on a Dense Kernel (i-j-k vs. i-k-j)

A and B are N x N float64 matrices stored as flat row-major 1D buffers
(leading dimension ld = N), where N is the salted dimension from
results/salted_params.json. Both algorithms compute the M x M sub-block
C[0:M, 0:M] = A[0:M, 0:M] @ B[0:M, 0:M] with M = N / 2. Because the
sub-block keeps row stride N, walking down a column of B moves N elements
(8N bytes) per step, exactly as in the full problem.

  Algorithm A (naive i-j-k):        for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
      inner loop reads B down a column -> stride N, a new cache line per step
  Algorithm B (interchanged i-k-j):  for i: for k: a = A[i,k]; for j: C[i,j] += a * B[k,j]
      A[i,k] hoisted into a register; inner loop streams B[k,:] and C[i,:]
      with stride 1 (8 elements per cache line, prefetch- and SIMD-friendly)

Each algorithm runs 3 times; the median wall-clock time is reported along with
the speedup S = T_naive / T_interchanged. Results are verified against NumPy.

Engines
  numba  (default when installed) -- both loop nests JIT-compiled with identical
         settings (strict IEEE, no fastmath); compilation happens before timing.
  python -- the same loops in pure CPython on flat lists. Interpreter overhead
         (~50-100 ns per step) dominates, so the measured speedup is small.

Usage (from the project root):
    python src/dense_kernel.py
    python src/dense_kernel.py --engine python
    python src/dense_kernel.py --student-id 12345678     # if salted_params.json is missing
Output:
    results/loop_interchange.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS = 3


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

# --------------------------------------------------------------------------- #
# Kernels -- flat row-major buffers, element (r, c) lives at r * ld + c
# --------------------------------------------------------------------------- #

def matmul_ijk(A, B, C, M, ld):
    """Algorithm A, naive i-j-k: inner loop walks B down a column (stride ld)."""
    for i in range(M):
        row_i = i * ld
        for j in range(M):
            acc = C[row_i + j]
            kj = j                              # index of B[k, j], starts at k = 0
            for k in range(M):
                acc += A[row_i + k] * B[kj]
                kj += ld                        # next row of B: jump ld elements
            C[row_i + j] = acc


def matmul_ikj(A, B, C, M, ld):
    """Algorithm B, interchanged i-k-j: A[i,k] hoisted, stride-1 rows of B and C."""
    for i in range(M):
        row_i = i * ld
        for k in range(M):
            a_ik = A[row_i + k]                 # register-local scalar
            row_k = k * ld
            for j in range(M):
                C[row_i + j] += a_ik * B[row_k + j]


def make_kernels(engine: str):
    if engine == "numba":
        import numba
        jit = numba.njit(cache=False, fastmath=False, nogil=True)
        ijk, ikj = jit(matmul_ijk), jit(matmul_ikj)
        # Trigger compilation outside the timed region.
        t = np.ones(16)
        ijk(t, t, np.zeros(16), 4, 4)
        ikj(t, t, np.zeros(16), 4, 4)
        return ijk, ikj, f"numba {numba.__version__} (njit, fastmath=False)"
    return matmul_ijk, matmul_ikj, f"pure CPython {platform.python_version()} (flat lists)"

# --------------------------------------------------------------------------- #
# Cache analysis of the naive column walk
# --------------------------------------------------------------------------- #

def column_walk_analysis(N: int, M: int, host: dict) -> dict:
    """How many L1d sets the naive inner loop's column of B maps onto."""
    line = host.get("cache_summary", {}).get("cache_line_bytes") or 64
    stride_bytes = N * 8
    info = {
        "b_column_stride_elements": N,
        "b_column_stride_bytes": stride_bytes,
        "cache_lines_per_column_walk": M,
        "column_walk_footprint_bytes": M * line,
    }
    l1 = host.get("caches", {}).get("L1d", {})
    size, assoc = l1.get("size_bytes"), l1.get("associativity")
    if size and isinstance(assoc, int) and assoc > 0:
        sets = size // (assoc * line)
        line_stride = stride_bytes // line
        sets_used = min(M, sets // math.gcd(line_stride, sets))
        info.update({
            "l1d_bytes": size,
            "l1d_associativity": assoc,
            "l1d_sets": sets,
            "l1d_sets_used_by_column": sets_used,
            "l1d_lines_available_to_column": sets_used * assoc,
            "column_fits_in_l1d": sets_used * assoc >= M,
        })
    return info

# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #

def time_kernel(kernel, A, B, N, M, runs, as_list):
    times, C = [], None
    for _ in range(runs):
        C = [0.0] * (N * N) if as_list else np.zeros(N * N)
        t0 = time.perf_counter_ns()
        kernel(A, B, C, M, N)
        times.append(time.perf_counter_ns() - t0)
    result = np.asarray(C, dtype=np.float64).reshape(N, N)[:M, :M]
    return times, result


def summarize(times_ns, M):
    median = statistics.median(times_ns)
    return {
        "times_s": [round(t / 1e9, 6) for t in times_ns],
        "median_s": round(median / 1e9, 6),
        "gflops": round(2 * M ** 3 / median, 4),      # flop / ns == GFLOP/s
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loop interchange benchmark (Task 1.2)")
    parser.add_argument("--engine", choices=["auto", "numba", "python"], default="auto")
    parser.add_argument("--student-id", help="8-digit Student ID (if salted_params.json is missing)")
    parser.add_argument("--m", type=int, default=None, help="override sub-block size (default N/2)")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "loop_interchange.json")
    args = parser.parse_args(argv)

    salted = _load_json(RESULTS_DIR / "salted_params.json")
    if args.student_id:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from salt import get_student_params, validate_student_id
        salted = get_student_params(validate_student_id(args.student_id))
    if not salted.get("N"):
        parser.error("no salted N: run `python src/profile_env.py --student-id <ID>` "
                     "first, or pass --student-id")
    host = _load_json(RESULTS_DIR / "host_profile.json")

    N, seed = int(salted["N"]), int(salted["seed"])
    M = args.m or N // 2

    engine = args.engine
    if engine == "auto":
        try:
            import numba  # noqa: F401
            engine = "numba"
        except ImportError:
            engine = "python"
            print("note: numba not installed -> pure Python (slow, small speedup). "
                  "Install with: pip install numba")
    ijk, ikj, engine_desc = make_kernels(engine)

    rng = np.random.default_rng(seed)
    A_np = rng.random(N * N)                 # flat row-major N x N buffers
    B_np = rng.random(N * N)
    as_list = engine == "python"
    A, B = (A_np.tolist(), B_np.tolist()) if as_list else (A_np, B_np)

    print(f"Engine   : {engine_desc}")
    print(f"Matrices : N = {N} (seed {seed}), sub-block M = {M}, ld = {N}")
    print(f"Work     : {M ** 3:,} multiply-adds per run, {args.runs} runs each")

    print("Running Algorithm A (naive i-j-k) ...", flush=True)
    t_naive, C_naive = time_kernel(ijk, A, B, N, M, args.runs, as_list)
    print("Running Algorithm B (interchanged i-k-j) ...", flush=True)
    t_inter, C_inter = time_kernel(ikj, A, B, N, M, args.runs, as_list)

    ref = A_np.reshape(N, N)[:M, :M] @ B_np.reshape(N, N)[:M, :M]
    err_naive = float(np.max(np.abs(C_naive - ref)))
    err_inter = float(np.max(np.abs(C_inter - ref)))
    tol = 1e-9 * M
    passed = err_naive <= tol and err_inter <= tol

    naive, inter = summarize(t_naive, M), summarize(t_inter, M)
    speedup = statistics.median(t_naive) / statistics.median(t_inter)

    result = {
        "task": "1.2 loop interchange (i-j-k vs i-k-j)",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "student_id": salted.get("student_id"),
        "N": N,
        "M": M,
        "leading_dimension": N,
        "seed": seed,
        "dtype": "float64",
        "layout": "flat row-major 1D buffer",
        "engine": engine_desc,
        "runs": args.runs,
        "statistic": "median",
        "naive_ijk": naive,
        "interchanged_ikj": inter,
        "naive_time_s": naive["median_s"],
        "interchanged_time_s": inter["median_s"],
        "speedup": round(speedup, 3),
        "speedup_definition": "S = T_naive / T_interchanged (median wall-clock)",
        "verification": {
            "reference": "numpy A[:M,:M] @ B[:M,:M]",
            "max_abs_error_naive": err_naive,
            "max_abs_error_interchanged": err_inter,
            "tolerance": tol,
            "passed": passed,
        },
        "column_walk_analysis": column_walk_analysis(N, M, host),
        "host_fingerprint": host.get("fingerprint", {}).get("salt"),
        "cpu": host.get("cpu", {}).get("model_name"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"{'algorithm':<26}{'median s':>10}{'GFLOP/s':>10}   runs (s)")
    print(f"{'A  naive i-j-k':<26}{naive['median_s']:>10.4f}{naive['gflops']:>10.3f}   {naive['times_s']}")
    print(f"{'B  interchanged i-k-j':<26}{inter['median_s']:>10.4f}{inter['gflops']:>10.3f}   {inter['times_s']}")
    print(f"\nSpeedup S = T_naive / T_interchanged = {speedup:.2f}x")
    print(f"Verification vs NumPy: {'PASSED' if passed else 'FAILED'} "
          f"(max abs error {max(err_naive, err_inter):.2e})")
    print(f"Wrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())