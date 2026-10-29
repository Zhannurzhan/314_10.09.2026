#!/usr/bin/env python3
"""
false_sharing.py -- Tasks 2.1-2.3: false sharing, remediation, Karp-Flatt diagnosis

Three variants, each run with P in {1, 2, 4, 8} worker processes, every worker
performing K = 5,000,000 increments:

  unpadded    (Task 2.1)  worker p increments shared_arr[p]. Indices 0..7 of a
                          64-byte-aligned int64 array share ONE cache line, so
                          every write invalidates the line in the other cores
                          (cache-line ping-pong).
  padded      (Task 2.2)  worker p increments shared_arr[p * 8]: each counter
                          sits on its own 64-byte cache line.
  privatized  (Task 2.2)  worker p counts in a local variable and performs a
                          single locked write to shared_arr[p] at the end.

Task 2.3 -- Karp-Flatt metric for P in {2, 4, 8}:

      e(P) = (1/S_P - 1/P) / (1 - 1/P)

Speedup is defined on the work actually done. By default each worker does K
increments (weak scaling: total work P*K grows with P), so
      S_P = (P*K / T_P) / (K / T_1) = P * T_1 / T_P   ->   e(P) = (T_P/T_1 - 1) / (P - 1)
With --scaling strong, K is split across the workers (K/P each), so
      S_P = T_1 / T_P                                  ->   e(P) = (P*T_P/T_1 - 1) / (P - 1)
Either way e = 0 means perfect scaling and growing e means growing overhead.

Implementation notes
  * Shared memory is mp.Array(ctypes.c_int64, n, lock=False). The default
    lock=True wraps every element access in one shared lock, which would
    measure lock contention instead of false sharing.
  * The counter block is aligned to a cache-line boundary inside the array, so
    indices 0..7 provably share one line, and guard space keeps other data off
    those lines.
  * Workers start together at a barrier and time only their loop; T(P) is the
    time from the first worker starting to the last worker finishing, so
    process-spawn cost (large on Windows) is excluded.
  * Each configuration runs --repeats times (default 3); the median is used.

Usage (from the project root):
    python src/false_sharing.py
    python src/false_sharing.py --iterations 1000000 --repeats 5
    python src/false_sharing.py --scaling strong
Outputs:
    results/false_sharing_karp_flatt.tsv
    results/false_sharing_karp_flatt.meta.json
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as _dt
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESS_COUNTS = [1, 2, 4, 8]
DEFAULT_K = 5_000_000
INT64 = ctypes.sizeof(ctypes.c_int64)
VARIANTS = ["unpadded", "padded", "privatized"]
BARRIER_TIMEOUT_S = 120      # workers give up if a sibling never reaches the barrier
RUN_TIMEOUT_S = 900          # parent terminates workers that run longer than this


# --------------------------------------------------------------------------- #
# Workers (top level so Windows' spawn start method can import them)
# --------------------------------------------------------------------------- #

def worker_unpadded(counter_array, worker_id, iterations, base, barrier, times):
    # Pathological: adjacent int64 counters, indices 0..7 on the SAME cache line.
    idx = base + worker_id
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        counter_array[idx] += 1
    t1 = time.perf_counter_ns()
    times[2 * worker_id], times[2 * worker_id + 1] = t0, t1


def worker_padded(counter_array, worker_id, iterations, base, stride, barrier, times):
    # Remedy 1: counters `stride` int64s (one cache line) apart.
    idx = base + worker_id * stride
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        counter_array[idx] += 1
    t1 = time.perf_counter_ns()
    times[2 * worker_id], times[2 * worker_id + 1] = t0, t1


def worker_privatized(counter_array, worker_id, iterations, base, lock, barrier, times):
    # Remedy 2: private accumulator, one synchronized write at the end.
    barrier.wait()
    t0 = time.perf_counter_ns()
    local = 0
    for _ in range(iterations):
        local += 1
    with lock:
        counter_array[base + worker_id] += local
    t1 = time.perf_counter_ns()
    times[2 * worker_id], times[2 * worker_id + 1] = t0, t1


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _aligned_base(arr, line_bytes: int) -> int:
    """Index of the first element of `arr` that starts a cache line."""
    misalign = ctypes.addressof(arr) % line_bytes
    return ((line_bytes - misalign) % line_bytes) // INT64


def run_once(ctx, variant: str, P: int, iters: int, line_bytes: int) -> float:
    line_elems = line_bytes // INT64                     # 8 for 64-byte lines
    used = P * line_elems if variant == "padded" else max(P, line_elems)
    # One spare line before (for alignment) and after (guard).
    shared_arr = ctx.Array(ctypes.c_int64, used + 2 * line_elems, lock=False)
    base = _aligned_base(shared_arr, line_bytes)
    times = ctx.Array(ctypes.c_int64, 2 * P, lock=False)
    barrier = ctx.Barrier(P, timeout=BARRIER_TIMEOUT_S)
    write_lock = ctx.Lock()          # one lock shared by all privatized workers

    procs = []
    for p in range(P):
        if variant == "unpadded":
            target, args = worker_unpadded, (shared_arr, p, iters, base, barrier, times)
        elif variant == "padded":
            target, args = worker_padded, (shared_arr, p, iters, base, line_elems, barrier, times)
        else:
            target, args = worker_privatized, (shared_arr, p, iters, base, write_lock, barrier, times)
        procs.append(ctx.Process(target=target, args=args))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=RUN_TIMEOUT_S)
    hung = [proc for proc in procs if proc.is_alive()]
    for proc in hung:
        proc.terminate()
        proc.join()
    if hung or any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError(f"{variant} P={P}: a worker failed or timed out")

    # Verify every counter and that nothing else was touched.
    stride = line_elems if variant == "padded" else 1
    expected = {base + p * stride for p in range(P)}
    for i in range(len(shared_arr)):
        want = iters if i in expected else 0
        if shared_arr[i] != want:
            raise RuntimeError(f"{variant} P={P}: shared_arr[{i}] = {shared_arr[i]}, expected {want}")

    starts, ends = times[0::2], times[1::2]
    return (max(ends) - min(starts)) / 1e9


def karp_flatt(speedup: float, P: int) -> float | None:
    if P <= 1 or speedup <= 0:
        return None
    return (1 / speedup - 1 / P) / (1 - 1 / P)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


TSV_COLUMNS = [
    "variant", "P", "scaling", "iterations_per_worker", "total_increments",
    "median_time_s", "min_time_s", "max_time_s", "samples_s",
    "ns_per_increment_per_worker", "throughput_Mincr_per_s",
    "time_ratio_vs_P1", "speedup", "efficiency", "karp_flatt_e", "oversubscribed",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="False sharing + Karp-Flatt (Tasks 2.1-2.3)")
    parser.add_argument("--iterations", "-K", type=int, default=DEFAULT_K,
                        help="increments per worker (default 5,000,000)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, nargs="+", default=PROCESS_COUNTS)
    parser.add_argument("--scaling", choices=["weak", "strong"], default="weak",
                        help="weak: K per worker (spec default); strong: K split across workers")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "false_sharing_karp_flatt.tsv")
    args = parser.parse_args(argv)

    host = _load_json(RESULTS_DIR / "host_profile.json")
    line_bytes = host.get("cache_summary", {}).get("cache_line_bytes") or 64
    physical = host.get("cpu", {}).get("physical_cores")
    logical = os.cpu_count()
    ctx = mp.get_context("spawn")       # identical behaviour on Windows, Linux, macOS
    K = args.iterations

    print(f"Python {platform.python_version()} on {platform.system()}, "
          f"{physical or '?'} physical / {logical} logical CPUs, {line_bytes}-byte lines")
    print(f"K = {K:,} increments, {args.scaling} scaling, {args.repeats} repeats, "
          f"P = {args.processes}")
    if 1 not in args.processes:
        parser.error("P = 1 is required as the baseline for speedup and Karp-Flatt")
    if max(args.processes) > logical:
        print(f"warning: P = {max(args.processes)} exceeds {logical} logical CPUs; "
              "those runs are oversubscribed")

    results: dict[tuple[str, int], list[float]] = {}
    for variant in VARIANTS:
        for P in args.processes:
            iters = K if args.scaling == "weak" else K // P
            samples = [run_once(ctx, variant, P, iters, line_bytes) for _ in range(args.repeats)]
            results[(variant, P)] = samples
            print(f"  {variant:<10} P={P}: median {statistics.median(samples):7.3f} s  "
                  f"(runs: {', '.join(f'{s:.3f}' for s in samples)})", flush=True)

    rows = []
    for variant in VARIANTS:
        t1 = statistics.median(results[(variant, 1)])
        for P in args.processes:
            samples = results[(variant, P)]
            tp = statistics.median(samples)
            iters = K if args.scaling == "weak" else K // P
            total = iters * P
            speedup = (P * t1 / tp) if args.scaling == "weak" else (t1 / tp)
            e = karp_flatt(speedup, P)
            rows.append({
                "variant": variant, "P": P, "scaling": args.scaling,
                "iterations_per_worker": iters, "total_increments": total,
                "median_time_s": f"{tp:.6f}", "min_time_s": f"{min(samples):.6f}",
                "max_time_s": f"{max(samples):.6f}",
                "samples_s": ",".join(f"{s:.6f}" for s in samples),
                "ns_per_increment_per_worker": f"{tp * 1e9 / iters:.2f}",
                "throughput_Mincr_per_s": f"{total / tp / 1e6:.3f}",
                "time_ratio_vs_P1": f"{tp / t1:.3f}",
                "speedup": f"{speedup:.4f}",
                "efficiency": f"{speedup / P:.4f}",
                "karp_flatt_e": "" if e is None else f"{e:.4f}",
                "oversubscribed": str(P > (physical or logical)).lower(),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'variant':<11}{'P':>3}{'median s':>10}{'T_P/T_1':>9}{'speedup':>9}{'e(P)':>9}")
    for r in rows:
        print(f"{r['variant']:<11}{r['P']:>3}{float(r['median_time_s']):>10.3f}"
              f"{float(r['time_ratio_vs_P1']):>9.2f}{float(r['speedup']):>9.2f}"
              f"{r['karp_flatt_e'] or '-':>9}")

    def e_of(variant, P):
        v = next(r["karp_flatt_e"] for r in rows if r["variant"] == variant and r["P"] == P)
        return float(v) if v else None

    p_max = max(args.processes)
    summary = {v: {str(P): e_of(v, P) for P in args.processes if P > 1} for v in VARIANTS}
    meta = {
        "tasks": "2.1-2.3 false sharing, padding/privatization, Karp-Flatt",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "iterations_per_worker_K": K,
        "scaling": args.scaling,
        "speedup_definition": ("S_P = P*T_1/T_P (each worker does K increments)"
                               if args.scaling == "weak" else
                               "S_P = T_1/T_P (K split across P workers)"),
        "karp_flatt": "e(P) = (1/S_P - 1/P) / (1 - 1/P)",
        "repeats": args.repeats,
        "statistic": "median",
        "timing": "max(worker end) - min(worker start), after a start barrier",
        "start_method": "spawn",
        "shared_memory": "multiprocessing.Array(ctypes.c_int64, n, lock=False), cache-line aligned",
        "cache_line_bytes": line_bytes,
        "physical_cores": physical,
        "logical_cpus": logical,
        "karp_flatt_by_variant": summary,
        "host_fingerprint": host.get("fingerprint", {}).get("salt"),
        "cpu": host.get("cpu", {}).get("model_name"),
        "python": platform.python_version(),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    eu, ep = e_of("unpadded", p_max), e_of("padded", p_max)
    if eu is not None and ep is not None:
        print(f"\nKarp-Flatt at P={p_max}: unpadded e = {eu:.3f}, padded e = {ep:.3f}, "
              f"privatized e = {e_of('privatized', p_max):.3f}")
    print(f"Wrote {args.output}\nWrote {meta_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())