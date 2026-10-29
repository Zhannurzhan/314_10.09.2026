#!/usr/bin/env python3
"""
stride_bench.py -- Task 1.1: Stride Latency Cliff Benchmark

Allocates a contiguous float64 array of 2^24 elements (128 MiB, ~134 MB),
far larger than any L3 cache, and for each stride s in {1,2,4,8,16,32,64,128}
traverses it with a reduction `total += data[i]` (i = 0, s, 2s, ...).

Per stride: 5 warm-up iterations, then 5 recorded iterations; the median
wall-clock time is logged and normalised per element read:

    latency per element (ns) = median elapsed time (ns) / elements touched

Expected shape (the "cache-line cliff"): with 64-byte lines, stride 1 gets
8 elements per line fetched from DRAM, so per-element cost is low. From
stride 8 (64 B) upward every read needs its own cache line, so the
per-element cost jumps by roughly the elements-per-line factor (~8x).

Methodology notes
  * Caches are flushed before every iteration (warm-up and recorded) by
    reading an eviction buffer larger than L3. Without this, large strides
    touch few enough lines to stay resident in L3 (stride 128 touches only
    8 MiB of lines) and would look artificially fast.
  * The reduction result is checked against the exact expected value so the
    work cannot be skipped and correctness is verified.
  * Engines: 'numpy' (default) sums a strided view, data[::s].sum(), in
    compiled code. 'numba' JIT-compiles the literal loop. 'python' runs the
    loop in the interpreter -- interpreter overhead (~50 ns/element) hides
    the memory effect, so it is only useful as a contrast.

Usage (from the project root):
    python src/stride_bench.py
    python src/stride_bench.py --engine numba
    python src/stride_bench.py --evict-mb 256
Outputs:
    results/stride_results.tsv       one row per stride
    results/stride_results.meta.json run configuration and summary
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_ELEMENTS = 16_777_216          # 2^24 float64 = 128 MiB
WARMUP_ITERS = 5
RECORDED_ITERS = 5
ELEMENT_BYTES = np.dtype(np.float64).itemsize


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _eviction_bytes(host: dict, override_mb: float | None) -> tuple[int, str]:
    """Pick an eviction buffer comfortably larger than the last-level cache."""
    if override_mb:
        return int(override_mb * (1 << 20)), "--evict-mb"
    caches = host.get("caches", {})
    llc = (caches.get("L3") or caches.get("L2") or {}).get("total_bytes")
    if llc:
        size = max(4 * llc, 64 << 20)
        return min(size, 1 << 30), "4 x LLC total from host_profile.json (64 MiB-1 GiB)"
    return 256 << 20, "default 256 MiB (host_profile.json not found)"


def _raise_priority() -> bool:
    """Best-effort: run at high priority to reduce scheduler noise."""
    try:
        import psutil  # optional
        proc = psutil.Process()
        if os.name == "nt":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(-5)
        return True
    except Exception:  # noqa: BLE001 -- psutil missing or not permitted
        return False


# --------------------------------------------------------------------------- #
# Kernels: each returns sum(data[0::s])
# --------------------------------------------------------------------------- #

def make_kernel(engine: str):
    if engine == "numpy":
        def numpy_kernel(data: np.ndarray, s: int) -> float:
            return float(data[::s].sum())
        return numpy_kernel, f"numpy {np.__version__}"

    if engine == "numba":
        import numba

        @numba.njit(fastmath=True, nogil=True, cache=False)
        def numba_kernel(a, s):
            total = 0.0
            for i in range(0, a.shape[0], s):
                total += a[i]
            return total

        numba_kernel(np.ones(64), 1)    # JIT-compile outside the timed region
        return numba_kernel, f"numba {numba.__version__}"

    if engine == "python":
        def python_kernel(data: np.ndarray, s: int) -> float:
            total = 0.0
            for i in range(0, data.shape[0], s):
                total += data[i]
            return float(total)
        return python_kernel, f"CPython loop {platform.python_version()}"

    raise ValueError(f"unknown engine {engine!r}")


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

def run_benchmark(kernel, data: np.ndarray, strides, line_bytes: int,
                  evict, warmup: int, repeats: int) -> list[dict]:
    n = data.shape[0]
    total_lines = -(-n * ELEMENT_BYTES // line_bytes)
    rows = []
    for s in strides:
        touched = -(-n // s)                       # ceil(n / s)
        stride_bytes = s * ELEMENT_BYTES
        lines = touched if stride_bytes >= line_bytes else total_lines
        expected = float(touched)                  # data is all ones

        samples = []
        for it in range(warmup + repeats):
            evict()
            gc_enabled = gc.isenabled()
            gc.disable()
            t0 = time.perf_counter_ns()
            total = kernel(data, s)
            t1 = time.perf_counter_ns()
            if gc_enabled:
                gc.enable()
            if total != expected:
                raise RuntimeError(f"stride {s}: sum {total} != expected {expected}")
            if it >= warmup:
                samples.append(t1 - t0)

        median_ns = statistics.median(samples)
        rows.append({
            "stride": s,
            "stride_bytes": stride_bytes,
            "elements_touched": touched,
            "cache_lines_touched": lines,
            "median_time_ns": int(median_ns),
            "min_time_ns": min(samples),
            "max_time_ns": max(samples),
            "latency_per_element_ns": median_ns / touched,
            "time_per_cache_line_ns": median_ns / lines,
            "dram_bandwidth_gbps": lines * line_bytes / median_ns,       # bytes/ns == GB/s
            "useful_bandwidth_gbps": touched * ELEMENT_BYTES / median_ns,
            "samples_ns": ",".join(str(x) for x in samples),
        })
        print(f"  stride {s:>3}: median {median_ns / 1e6:9.3f} ms, "
              f"{median_ns / touched:7.3f} ns/element", flush=True)

    base = rows[0]["latency_per_element_ns"]
    for row in rows:
        row["slowdown_vs_stride1"] = row["latency_per_element_ns"] / base
    return rows


TSV_COLUMNS = [
    "stride", "stride_bytes", "elements_touched", "cache_lines_touched",
    "median_time_ns", "min_time_ns", "max_time_ns",
    "latency_per_element_ns", "time_per_cache_line_ns", "slowdown_vs_stride1",
    "dram_bandwidth_gbps", "useful_bandwidth_gbps", "samples_ns",
]


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("latency_per_element_ns", "time_per_cache_line_ns",
                        "slowdown_vs_stride1", "dram_bandwidth_gbps",
                        "useful_bandwidth_gbps"):
                out[key] = f"{row[key]:.4f}"
            writer.writerow(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stride latency cliff benchmark (Task 1.1)")
    parser.add_argument("--engine", choices=["numpy", "numba", "python"], default="numpy")
    parser.add_argument("--elements", type=int, default=None,
                        help="array length (default: buffer_elements from "
                             "results/salted_params.json, else 2^24)")
    parser.add_argument("--evict-mb", type=float, default=None,
                        help="cache-eviction buffer size in MiB (default: 4 x L3)")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--repeats", type=int, default=RECORDED_ITERS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "stride_results.tsv")
    args = parser.parse_args(argv)

    host = _load_json(RESULTS_DIR / "host_profile.json")
    salted = _load_json(RESULTS_DIR / "salted_params.json")
    n = args.elements or salted.get("buffer_elements") or DEFAULT_ELEMENTS
    line_bytes = host.get("cache_summary", {}).get("cache_line_bytes") or 64
    evict_bytes, evict_src = _eviction_bytes(host, args.evict_mb)
    kernel, engine_desc = make_kernel(args.engine)
    priority = _raise_priority()

    print(f"Engine        : {engine_desc}")
    print(f"Array         : {n:,} x float64 = {n * ELEMENT_BYTES / 2**20:.0f} MiB")
    print(f"Cache line    : {line_bytes} B ({line_bytes // ELEMENT_BYTES} elements/line)")
    print(f"Evict buffer  : {evict_bytes / 2**20:.0f} MiB ({evict_src})")
    print(f"Iterations    : {args.warmup} warm-up + {args.repeats} recorded per stride")
    if not host:
        print("note: results/host_profile.json not found -- run src/profile_env.py first")

    # Allocate flat array in physical memory (np.ones writes every page).
    data = np.ones(n, dtype=np.float64)
    evict_buf = np.ones(evict_bytes // ELEMENT_BYTES, dtype=np.float64)

    def evict() -> None:
        evict_buf.sum()                 # stream through > LLC to flush `data`

    rows = run_benchmark(kernel, data, STRIDES, line_bytes, evict,
                         args.warmup, args.repeats)
    write_tsv(rows, args.output)

    by_stride = {r["stride"]: r for r in rows}
    cliff = by_stride[8]["latency_per_element_ns"] / by_stride[1]["latency_per_element_ns"]
    print()
    print(f"{'stride':>6} {'bytes':>6} {'touched':>10} {'median ms':>10} "
          f"{'ns/elem':>8} {'x s=1':>6} {'GB/s DRAM':>9}")
    for r in rows:
        print(f"{r['stride']:>6} {r['stride_bytes']:>6} {r['elements_touched']:>10,} "
              f"{r['median_time_ns'] / 1e6:>10.3f} {r['latency_per_element_ns']:>8.3f} "
              f"{r['slowdown_vs_stride1']:>6.2f} {r['dram_bandwidth_gbps']:>9.2f}")
    print(f"\nCache-line cliff: stride 8 costs {cliff:.1f}x more per element than "
          f"stride 1 (ideal ~{line_bytes // ELEMENT_BYTES}x).")
    if cliff < 2:
        print("warning: weak cliff -- close other programs, plug in the charger, "
              "and re-run; or try --engine numba.")

    meta = {
        "task": "1.1 stride latency cliff",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": engine_desc,
        "elements": n,
        "array_bytes": n * ELEMENT_BYTES,
        "element_bytes": ELEMENT_BYTES,
        "cache_line_bytes": line_bytes,
        "strides": STRIDES,
        "warmup_iterations": args.warmup,
        "recorded_iterations": args.repeats,
        "statistic": "median",
        "timer": "time.perf_counter_ns",
        "eviction_buffer_bytes": evict_bytes,
        "eviction_source": evict_src,
        "high_priority": priority,
        "cliff_ratio_stride8_vs_stride1": round(cliff, 3),
        "student_id": salted.get("student_id"),
        "host_fingerprint": host.get("fingerprint", {}).get("salt"),
        "cpu": host.get("cpu", {}).get("model_name"),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}\nWrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())