#!/usr/bin/env python3
"""
scaling_study.py -- Milestone 3: strong scaling (3.1), weak scaling (3.2), roofline (3.3)

Kernel: 5-point Jacobi stencil on an N x N float64 grid (fixed boundaries),
    v[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
repeated for `sweeps` iterations with u/v swapped after each sweep.
Work is 4 FLOPs (3 adds + 1 multiply) per interior point per sweep.

Parallelisation: P spawned worker processes; the two grids live in
multiprocessing.shared_memory. Interior rows are split into P contiguous
blocks; each worker updates its block with vectorised NumPy and all workers
meet at a barrier after every sweep (halo rows are read straight from the
neighbours' shared rows). Workers start together at a barrier and time only
the sweeps, so process-spawn cost is excluded. T_P is the median over
--repeats runs of (last worker end - first worker start).

Task 3.1  strong scaling, fixed N = N_salted, P in {1, 2, 4, ..., physical cores}:
          S_P = T_1 / T_P,   E_P = S_P / P * 100 %
          serial fraction per P:  s(P) = (1/S_P - 1/P) / (1 - 1/P)
          least-squares Amdahl fit of s over all P, and S_max = 1 / s
Task 3.2  weak scaling, N_P = ceil(N * sqrt(P)) so work grows as O(P):
          S_P^scaled = P * Work(1) * T_1 / (Work(P) * T_P^weak)
          Gustafson serial fraction:  s' = (P - S_P^scaled) / (P - 1)
Task 3.3  arithmetic intensity, machine balance and roofline regime, using
          cache sizes from results/host_profile.json, a theoretical peak
          memory bandwidth (WMI on Windows, or --peak-bw-gbs), and a measured
          STREAM-Add bandwidth (1 process and all physical cores).

Usage (from the project root):
    python src/scaling_study.py
    python src/scaling_study.py --sweeps 200 --repeats 5
    python src/scaling_study.py --peak-bw-gbs 89.6        # e.g. DDR5-5600 dual channel
Outputs:
    results/scaling_metrics.tsv     one row per (study, P)
    results/scaling_summary.json    Amdahl / Gustafson fits and the contrast
    results/roofline_summary.json   AI, machine balance, regime conclusion
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FLOPS_PER_POINT = 4                  # 3 additions + 1 multiplication
BYTES_PER_POINT_IDEAL = 24           # read u (8) + write v (8) + write-allocate of v (8)
BYTES_PER_POINT_NO_WA = 16           # same with non-temporal / streaming stores
BYTES_PER_POINT_NUMPY = 96           # this NumPy implementation: 4 passes x ~24 B (see below)
BARRIER_TIMEOUT_S = 600
RUN_TIMEOUT_S = 3600


# --------------------------------------------------------------------------- #
# Stencil kernel and worker (top level so the spawn start method can import it)
# --------------------------------------------------------------------------- #

def init_rows(grid: np.ndarray, i0: int, i1: int, phase: float) -> None:
    """Deterministic initial field, independent of how rows are partitioned."""
    if i1 <= i0:
        return
    n = grid.shape[1]
    i = np.arange(i0, i1, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    grid[i0:i1, :] = np.sin(0.037 * i + phase) * np.cos(0.023 * j - phase)


def jacobi_sweep(src: np.ndarray, dst: np.ndarray, r0: int, r1: int, tmp: np.ndarray) -> None:
    """dst[r0:r1, 1:-1] = 0.25 * (north + south + west + east); 4 FLOPs per point."""
    np.add(src[r0 - 1:r1 - 1, 1:-1], src[r0 + 1:r1 + 1, 1:-1], out=tmp)
    np.add(tmp, src[r0:r1, :-2], out=tmp)
    np.add(tmp, src[r0:r1, 2:], out=tmp)
    np.multiply(tmp, 0.25, out=dst[r0:r1, 1:-1])


def _attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _stencil_compute(wid, P, bufs, N, r0, r1, sweeps, repeats, phase, barrier, times):
    grids = [np.ndarray((N, N), dtype=np.float64, buffer=b) for b in bufs]
    i0 = 0 if r0 == 1 else r0                  # first worker also owns boundary row 0
    i1 = N if r1 == N - 1 else r1              # last worker also owns boundary row N-1
    tmp = np.empty((r1 - r0, N - 2))
    for rep in range(repeats):
        for grid in grids:
            init_rows(grid, i0, i1, phase)
        barrier.wait()
        t0 = time.perf_counter_ns()
        src, dst = grids
        for _ in range(sweeps):
            jacobi_sweep(src, dst, r0, r1, tmp)
            barrier.wait()                     # neighbours' rows complete before next sweep
            src, dst = dst, src
        t1 = time.perf_counter_ns()
        k = 2 * (rep * P + wid)
        times[k], times[k + 1] = t0, t1


def stencil_worker(wid, P, names, N, r0, r1, sweeps, repeats, phase, barrier, times):
    shms = [_attach(n) for n in names]
    try:
        _stencil_compute(wid, P, [s.buf for s in shms], N, r0, r1,
                         sweeps, repeats, phase, barrier, times)
    finally:
        for s in shms:
            try:
                s.close()
            except BufferError:
                pass


def bandwidth_worker(wid, P, nelem, reps, barrier, times):
    a = np.full(nelem, 1.0)
    b = np.full(nelem, 2.0)
    c = np.empty(nelem)
    np.add(a, b, out=c)                        # touch all pages before timing
    for rep in range(reps):
        barrier.wait()
        t0 = time.perf_counter_ns()
        np.add(a, b, out=c)
        t1 = time.perf_counter_ns()
        k = 2 * (rep * P + wid)
        times[k], times[k + 1] = t0, t1


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def _run_processes(procs) -> None:
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=RUN_TIMEOUT_S)
    hung = [proc for proc in procs if proc.is_alive()]
    for proc in hung:
        proc.terminate()
        proc.join()
    if hung or any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError("a worker process failed or timed out")


def _spans(times, P: int, reps: int) -> list[int]:
    out = []
    for rep in range(reps):
        starts = [times[2 * (rep * P + p)] for p in range(P)]
        ends = [times[2 * (rep * P + p) + 1] for p in range(P)]
        out.append(max(ends) - min(starts))
    return out


def run_stencil(ctx, N: int, P: int, sweeps: int, repeats: int, phase: float,
                keep_grid: bool = False):
    """Return (samples_s, checksum, final_grid_or_None)."""
    shms = [shared_memory.SharedMemory(create=True, size=N * N * 8) for _ in range(2)]
    try:
        times = ctx.Array(ctypes.c_int64, 2 * P * repeats, lock=False)
        barrier = ctx.Barrier(P, timeout=BARRIER_TIMEOUT_S)
        edges = [1 + (N - 2) * p // P for p in range(P + 1)]
        names = [s.name for s in shms]
        procs = [ctx.Process(target=stencil_worker,
                             args=(p, P, names, N, edges[p], edges[p + 1],
                                   sweeps, repeats, phase, barrier, times))
                 for p in range(P)]
        _run_processes(procs)
        samples = [ns / 1e9 for ns in _spans(times, P, repeats)]

        final = shms[0] if sweeps % 2 == 0 else shms[1]
        view = np.ndarray((N, N), dtype=np.float64, buffer=final.buf)
        checksum = float(view.sum())
        grid = view.copy() if keep_grid else None
        del view
        return samples, checksum, grid
    finally:
        for s in shms:
            s.close()
            s.unlink()


def measure_bandwidth(ctx, P: int, llc_bytes: int, reps: int) -> dict:
    """STREAM-Add style c = a + b in P processes at once; counts 24 B/element."""
    nelem = max(math.ceil(4 * llc_bytes / (24 * P)), 1 << 20)
    times = ctx.Array(ctypes.c_int64, 2 * P * reps, lock=False)
    barrier = ctx.Barrier(P, timeout=BARRIER_TIMEOUT_S)
    procs = [ctx.Process(target=bandwidth_worker, args=(p, P, nelem, reps, barrier, times))
             for p in range(P)]
    _run_processes(procs)
    rates = [P * nelem * 24 / ns for ns in _spans(times, P, reps)]   # bytes/ns == GB/s
    return {
        "processes": P,
        "elements_per_process": nelem,
        "aggregate_bytes_per_rep": P * nelem * 24,
        "best_gbs": round(max(rates), 2),
        "median_gbs": round(statistics.median(rates), 2),
    }

# --------------------------------------------------------------------------- #
# Machine characteristics
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def theoretical_memory_bandwidth(override: float | None) -> dict:
    if override:
        return {"gbs": override, "source": "--peak-bw-gbs"}
    if platform.system() != "Windows":
        return {"gbs": None, "source": "unavailable on this OS; pass --peak-bw-gbs"}
    script = ("Get-CimInstance Win32_PhysicalMemory | Select-Object "
              "ConfiguredClockSpeed,Speed,DataWidth,Capacity | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=30, check=False).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"gbs": None, "source": "WMI query failed; pass --peak-bw-gbs"}
    modules = data if isinstance(data, list) else [data]
    modules = [m for m in modules if (m.get("ConfiguredClockSpeed") or m.get("Speed"))]
    if not modules:
        return {"gbs": None, "source": "WMI returned no memory modules; pass --peak-bw-gbs"}
    mts = max(m.get("ConfiguredClockSpeed") or m.get("Speed") for m in modules)
    width = max(int(m.get("DataWidth") or 64) for m in modules)
    channels = min(len(modules), 2)
    return {
        "gbs": round(mts * 1e6 * (width / 8) * channels / 1e9, 2),
        "transfer_rate_mts": mts,
        "bus_width_bits": width,
        "modules": len(modules),
        "channels_assumed": channels,
        "total_capacity_bytes": sum(int(m.get("Capacity") or 0) for m in modules),
        "source": ("WMI Win32_PhysicalMemory; estimate assumes one 64-bit channel per "
                   "module, at most 2 (consumer dual channel). Override with --peak-bw-gbs."),
    }


def default_flops_per_cycle(uarch: str) -> tuple[int, str]:
    u = (uarch or "").lower()
    if any(k in u for k in ("skylake-sp", "cascade", "ice lake-sp", "sapphire",
                            "emerald", "granite")):
        return 32, "AVX-512: 2 FMA units x 8 doubles x 2 FLOPs"
    if "zen 5" in u:
        return 32, "Zen 5 AVX-512 (desktop; mobile parts are 16)"
    if any(k in u for k in ("sandy", "ivy")):
        return 8, "AVX without FMA: 4-wide add + 4-wide mul"
    if "apple" in u:
        return 16, "4 x 128-bit FMA units x 2 doubles x 2 FLOPs"
    return 16, "AVX2: 2 FMA units x 4 doubles x 2 FLOPs"


def process_counts(physical: int) -> list[int]:
    counts, p = [], 1
    while p <= physical:
        counts.append(p)
        p *= 2
    if counts[-1] != physical:
        counts.append(physical)
    return counts

# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def serial_fraction(speedup: float, P: int) -> float | None:
    """Amdahl / Karp-Flatt: s = (1/S - 1/P) / (1 - 1/P)."""
    if P <= 1:
        return None
    return (1 / speedup - 1 / P) / (1 - 1 / P)


def fit_amdahl(points: list[tuple[int, float]]) -> float | None:
    """Least-squares s in 1/S - 1/P = s * (1 - 1/P) over all P > 1."""
    xs = [1 - 1 / P for P, _ in points if P > 1]
    ys = [1 / S - 1 / P for P, S in points if P > 1]
    if not xs:
        return None
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


TSV_COLUMNS = [
    "study", "P", "N", "interior_points", "sweeps", "work_point_updates",
    "median_time_s", "min_time_s", "max_time_s", "samples_s",
    "speedup", "speedup_definition", "efficiency_pct",
    "serial_fraction", "serial_fraction_definition",
    "mupdates_per_s", "gflops", "verified",
]


def _row(study, P, N, sweeps, samples, speedup, sp_def, sf, sf_def, verified):
    interior = (N - 2) ** 2
    t = statistics.median(samples)
    return {
        "study": study, "P": P, "N": N, "interior_points": interior, "sweeps": sweeps,
        "work_point_updates": interior * sweeps,
        "median_time_s": f"{t:.6f}", "min_time_s": f"{min(samples):.6f}",
        "max_time_s": f"{max(samples):.6f}",
        "samples_s": ",".join(f"{s:.6f}" for s in samples),
        "speedup": f"{speedup:.4f}", "speedup_definition": sp_def,
        "efficiency_pct": f"{speedup / P * 100:.2f}",
        "serial_fraction": "" if sf is None else f"{sf:.5f}",
        "serial_fraction_definition": sf_def if sf is not None else "",
        "mupdates_per_s": f"{interior * sweeps / t / 1e6:.2f}",
        "gflops": f"{FLOPS_PER_POINT * interior * sweeps / t / 1e9:.3f}",
        "verified": verified,
    }

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strong/weak scaling and roofline (Milestone 3)")
    parser.add_argument("--sweeps", type=int, default=100, help="stencil sweeps per run")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, nargs="+", default=None,
                        help="process counts (default 1, 2, 4, ... physical cores)")
    parser.add_argument("--student-id", help="8-digit Student ID if salted_params.json is missing")
    parser.add_argument("--peak-bw-gbs", type=float, default=None,
                        help="theoretical peak DRAM bandwidth in GB/s (overrides WMI estimate)")
    parser.add_argument("--flops-per-cycle", type=int, default=None,
                        help="peak FP64 FLOPs per cycle per core (default from microarchitecture)")
    parser.add_argument("--bw-reps", type=int, default=5, help="repetitions of the bandwidth test")
    parser.add_argument("--skip-weak", action="store_true")
    parser.add_argument("--skip-roofline", action="store_true")
    args = parser.parse_args(argv)

    host = _load_json(RESULTS_DIR / "host_profile.json")
    salted = _load_json(RESULTS_DIR / "salted_params.json")
    if args.student_id:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from salt import get_student_params, validate_student_id
        salted = get_student_params(validate_student_id(args.student_id))
    if not salted.get("N"):
        parser.error("no salted N: run `python src/profile_env.py --student-id <ID>` first")

    N, seed = int(salted["N"]), int(salted["seed"])
    phase = (seed % 1000) / 1000 * 2 * math.pi
    cpu = host.get("cpu", {})
    physical = cpu.get("physical_cores") or os.cpu_count()
    P_list = sorted(set(args.processes)) if args.processes else process_counts(physical)
    if 1 not in P_list:
        parser.error("P = 1 is required as the baseline")
    ctx = mp.get_context("spawn")

    print(f"Stencil  : 5-point Jacobi, N = {N} (salted), {args.sweeps} sweeps, "
          f"{args.repeats} repeats, P = {P_list}")
    print(f"CPU      : {cpu.get('model_name')} ({physical} physical cores)")

    # ---------------- Task 3.1: strong scaling ----------------
    print("\n[3.1] Strong scaling (fixed N)")
    strong, reference = {}, None
    for P in P_list:
        samples, checksum, grid = run_stencil(ctx, N, P, args.sweeps, args.repeats, phase,
                                              keep_grid=True)
        if P == 1:
            reference, ok = grid, True
        else:
            ok = bool(np.array_equal(grid, reference))
        del grid
        strong[P] = (samples, ok)
        print(f"  P={P:<3} median {statistics.median(samples):8.4f} s  "
              f"result matches P=1: {ok}", flush=True)
    reference = None
    if not all(ok for _, ok in strong.values()):
        print("ERROR: parallel result differs from the serial result")

    T1 = statistics.median(strong[1][0])
    rows, strong_points = [], []
    for P in P_list:
        samples, ok = strong[P]
        S = T1 / statistics.median(samples)
        strong_points.append((P, S))
        rows.append(_row("strong", P, N, args.sweeps, samples, S, "T_1/T_P",
                         serial_fraction(S, P), "Amdahl/Karp-Flatt (1/S-1/P)/(1-1/P)",
                         str(ok).lower()))
    s_fit = fit_amdahl(strong_points)
    s_clamped = None if s_fit is None else max(s_fit, 0.0)
    s_max = None if not s_clamped else 1 / s_clamped

    # ---------------- Task 3.2: weak scaling ----------------
    weak_points = []
    if not args.skip_weak:
        print("\n[3.2] Weak scaling (N_P = ceil(N * sqrt(P)))")
        work1 = (N - 2) ** 2
        for P in P_list:
            NP = math.ceil(N * math.sqrt(P))
            if P == 1:
                samples = strong[1][0]          # identical configuration, reuse
            else:
                samples, _, _ = run_stencil(ctx, NP, P, args.sweeps, args.repeats, phase)
            TP = statistics.median(samples)
            workP = (NP - 2) ** 2
            S_scaled = P * work1 * T1 / (workP * TP)
            weak_points.append((P, NP, S_scaled, TP))
            g = None if P == 1 else (P - S_scaled) / (P - 1)
            rows.append(_row("weak", P, NP, args.sweeps, samples, S_scaled,
                             "P*Work(1)*T_1/(Work(P)*T_P)", g,
                             "Gustafson (P-S)/(P-1)", ""))
            print(f"  P={P:<3} N_P={NP:<6} median {TP:8.4f} s  scaled speedup {S_scaled:6.2f}",
                  flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = RESULTS_DIR / "scaling_metrics.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'P':>4} {'strong S':>9} {'strong E%':>10} {'s(P)':>8}"
          + ("   " + f"{'N_P':>6} {'scaled S':>9} {'weak E%':>8}" if weak_points else ""))
    for i, (P, S) in enumerate(strong_points):
        sf = serial_fraction(S, P)
        line = f"{P:>4} {S:>9.2f} {S / P * 100:>10.1f} {('-' if sf is None else f'{sf:.4f}'):>8}"
        if weak_points:
            _, NP, Sw, _ = weak_points[i]
            line += f"   {NP:>6} {Sw:>9.2f} {Sw / P * 100:>8.1f}"
        print(line)
    if s_fit is not None:
        print(f"\nAmdahl fit: serial fraction s = {s_fit:.4f}  ->  S_max = 1/s = "
              f"{('unbounded' if not s_max else f'{s_max:.1f}x')}")

    P_top = P_list[-1]
    S_top = dict(strong_points)[P_top]
    summary = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "student_id": salted.get("student_id"),
        "kernel": "5-point Jacobi stencil, float64, NumPy row-block decomposition",
        "N": N, "sweeps": args.sweeps, "repeats": args.repeats, "statistic": "median",
        "process_counts": P_list,
        "strong_scaling": {
            "T1_s": round(T1, 6),
            "per_P": {str(P): {"speedup": round(S, 4), "efficiency_pct": round(S / P * 100, 2),
                               "serial_fraction": (None if P == 1
                                                   else round(serial_fraction(S, P), 5))}
                      for P, S in strong_points},
            "amdahl_fit": {
                "method": "least squares of 1/S - 1/P = s (1 - 1/P) over P > 1",
                "serial_fraction_s": None if s_fit is None else round(s_fit, 5),
                "max_speedup_S_max": None if not s_max else round(s_max, 2),
                "note": ("s <= 0 means superlinear or perfect scaling; S_max is unbounded"
                         if s_fit is not None and s_fit <= 0 else None),
            },
            "results_identical_to_serial": all(ok for _, ok in strong.values()),
        },
    }
    if weak_points:
        summary["weak_scaling"] = {
            "rule": "N_P = ceil(N * sqrt(P)), work ~ O(P)",
            "per_P": {str(P): {"N_P": NP, "time_s": round(TP, 6),
                               "scaled_speedup": round(Sw, 4),
                               "efficiency_pct": round(Sw / P * 100, 2),
                               "gustafson_serial_fraction": (None if P == 1
                                                             else round((P - Sw) / (P - 1), 5))}
                      for P, NP, Sw, TP in weak_points},
        }
        Sw_top = weak_points[-1][2]
        summary["contrast"] = {
            "P": P_top,
            "strong_efficiency_pct": round(S_top / P_top * 100, 2),
            "weak_efficiency_pct": round(Sw_top / P_top * 100, 2),
            "statement": (f"At P={P_top}, strong scaling reaches {S_top:.2f}x "
                          f"({S_top / P_top * 100:.1f}% efficiency) on the fixed N={N} grid, "
                          f"while weak scaling delivers a scaled speedup of {Sw_top:.2f}x "
                          f"({Sw_top / P_top * 100:.1f}% efficiency) when the grid grows "
                          f"to N={weak_points[-1][1]}."),
        }
    (RESULTS_DIR / "scaling_summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")

    # ---------------- Task 3.3: roofline ----------------
    if not args.skip_roofline:
        print("\n[3.3] Roofline: measuring memory bandwidth ...", flush=True)
        caches = host.get("caches", {})
        l1d = caches.get("L1d", {}).get("size_bytes")
        l2 = caches.get("L2", {}).get("size_bytes")
        l2_total = caches.get("L2", {}).get("total_bytes")
        l3_total = caches.get("L3", {}).get("total_bytes") or caches.get("L3", {}).get("size_bytes")
        llc = l3_total or l2_total or (32 << 20)

        bw_1 = measure_bandwidth(ctx, 1, llc, args.bw_reps)
        bw_all = measure_bandwidth(ctx, physical, llc, args.bw_reps) if physical > 1 else bw_1
        theo = theoretical_memory_bandwidth(args.peak_bw_gbs)

        uarch = cpu.get("microarchitecture", "")
        fpc, fpc_src = ((args.flops_per_cycle, "--flops-per-cycle") if args.flops_per_cycle
                        else default_flops_per_cycle(uarch))
        ghz = (cpu.get("base_frequency_mhz") or 0) / 1000
        peak_gflops = physical * ghz * fpc if ghz else None

        ai_ideal = FLOPS_PER_POINT / BYTES_PER_POINT_IDEAL
        ai_nowa = FLOPS_PER_POINT / BYTES_PER_POINT_NO_WA
        ai_impl = FLOPS_PER_POINT / BYTES_PER_POINT_NUMPY
        bw_ceiling = theo.get("gbs") or bw_all["best_gbs"]
        ridge_theo = peak_gflops / theo["gbs"] if peak_gflops and theo.get("gbs") else None
        ridge_meas = peak_gflops / bw_all["best_gbs"] if peak_gflops else None
        ridge = ridge_theo or ridge_meas
        attainable = min(peak_gflops, ai_ideal * bw_ceiling) if peak_gflops else ai_ideal * bw_ceiling

        ws_strong = 3 * N * N * 8                     # u, v and the per-worker temporaries
        best_P = min(P_list, key=lambda P: statistics.median(strong[P][0]))
        t_best = statistics.median(strong[best_P][0])
        updates = (N - 2) ** 2 * args.sweeps
        achieved_gflops = FLOPS_PER_POINT * updates / t_best / 1e9
        achieved_bw_impl = BYTES_PER_POINT_NUMPY * updates / t_best / 1e9

        bound = "Memory-Bandwidth Bound" if (ridge is None or ai_ideal < ridge) else "Compute Bound"
        if l3_total and ws_strong <= l3_total:
            residency = (f"The strong-scaling working set ({ws_strong / 2**20:.1f} MiB) fits in "
                         f"L3 ({l3_total / 2**20:.1f} MiB), so the binding ceiling is L3 "
                         f"bandwidth rather than DRAM; the kernel is still bandwidth bound.")
        else:
            residency = (f"The strong-scaling working set ({ws_strong / 2**20:.1f} MiB) exceeds "
                         f"L3 ({(l3_total or 0) / 2**20:.1f} MiB), so every sweep streams "
                         f"from DRAM.")
        if l2 and ws_strong / P_top <= l2:
            residency += (f" At P={P_top} each worker's share ({ws_strong / P_top / 2**20:.2f} MiB) "
                          f"fits in its private L2 ({l2 / 2**20:.2f} MiB), which can produce "
                          f"superlinear strong-scaling speedup.")

        roofline = {
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "kernel": {
                "name": "5-point Jacobi stencil (float64)",
                "update": "v[i,j] = 0.25*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])",
                "flops_per_point": FLOPS_PER_POINT,
                "flop_breakdown": "3 additions + 1 multiplication",
            },
            "arithmetic_intensity": round(ai_ideal, 4),
            "arithmetic_intensity_units": "FLOP/byte",
            "arithmetic_intensity_details": {
                "theoretical_flop_per_byte": round(ai_ideal, 4),
                "theoretical_bytes_per_point": BYTES_PER_POINT_IDEAL,
                "theoretical_assumptions": ("perfect reuse of the 3-row neighbourhood in cache: "
                                            "u read once (8 B), v written once (8 B) plus its "
                                            "write-allocate read (8 B)"),
                "streaming_stores_flop_per_byte": round(ai_nowa, 4),
                "as_implemented_flop_per_byte": round(ai_impl, 4),
                "as_implemented_bytes_per_point": BYTES_PER_POINT_NUMPY,
                "as_implemented_note": ("NumPy evaluates the update in 4 passes through a "
                                        "temporary (add, add, add, scale), each streaming "
                                        "~24 B per point when the block exceeds cache"),
            },
            "machine": {
                "cpu": cpu.get("model_name"),
                "microarchitecture": uarch,
                "physical_cores": physical,
                "base_frequency_ghz": round(ghz, 3) if ghz else None,
                "fp64_flops_per_cycle_per_core": fpc,
                "flops_per_cycle_source": fpc_src,
                "peak_fp64_gflops": round(peak_gflops, 1) if peak_gflops else None,
                "caches_bytes": {"L1d_per_core": l1d, "L2_per_instance": l2,
                                 "L2_total": l2_total, "L3_total": l3_total},
                "cache_line_bytes": host.get("cache_summary", {}).get("cache_line_bytes"),
                "peak_memory_bandwidth_theoretical": theo,
                "memory_bandwidth_measured": {
                    "kernel": "STREAM Add c = a + b, 24 B per element",
                    "single_process": bw_1,
                    "all_physical_cores": bw_all,
                },
            },
            "roofline": {
                "ridge_point_theoretical_flop_per_byte": None if ridge_theo is None else round(ridge_theo, 2),
                "ridge_point_measured_bw_flop_per_byte": None if ridge_meas is None else round(ridge_meas, 2),
                "bandwidth_ceiling_used_gbs": bw_ceiling,
                "attainable_gflops_at_theoretical_ai": round(attainable, 2),
                "achieved": {
                    "best_P": best_P,
                    "gflops": round(achieved_gflops, 3),
                    "implied_dram_traffic_gbs_as_implemented": round(achieved_bw_impl, 2),
                    "fraction_of_measured_all_core_bandwidth": round(
                        achieved_bw_impl / bw_all["best_gbs"], 3),
                    "fraction_of_peak_flops": (None if not peak_gflops
                                               else round(achieved_gflops / peak_gflops, 4)),
                },
            },
            "working_set": {
                "strong_N": N,
                "bytes": ws_strong,
                "fits_in_L3": None if not l3_total else ws_strong <= l3_total,
                "per_worker_bytes_at_max_P": ws_strong // P_top,
            },
            "regime": bound,
            "conclusion": (
                f"The stencil performs {FLOPS_PER_POINT} FLOPs per {BYTES_PER_POINT_IDEAL} bytes "
                f"(AI = {ai_ideal:.3f} FLOP/B; {ai_impl:.3f} FLOP/B as implemented in NumPy). "
                + (f"The machine balance (ridge point) is {ridge:.1f} FLOP/B, about "
                   f"{ridge / ai_ideal:.0f}x higher, " if ridge else "")
                + f"so the kernel is {bound.lower()}: attainable performance is capped near "
                  f"AI x bandwidth = {attainable:.1f} GFLOP/s"
                + (f", far below the {peak_gflops:.0f} GFLOP/s compute peak. " if peak_gflops else ". ")
                + residency
                + " Adding cores helps only until the shared memory bandwidth saturates, which "
                  "also limits strong-scaling efficiency."
            ),
        }
        (RESULTS_DIR / "roofline_summary.json").write_text(json.dumps(roofline, indent=2),
                                                           encoding="utf-8")
        print(f"  measured bandwidth: {bw_1['best_gbs']} GB/s (1 process), "
              f"{bw_all['best_gbs']} GB/s ({physical} processes); theoretical: "
              f"{theo.get('gbs') or 'n/a'} GB/s")
        print(f"  AI = {ai_ideal:.3f} FLOP/B (as implemented {ai_impl:.3f}); ridge point = "
              f"{('n/a' if ridge is None else f'{ridge:.2f}')} FLOP/B  ->  {bound}")

    print(f"\nWrote {tsv_path}")
    print(f"Wrote {RESULTS_DIR / 'scaling_summary.json'}")
    if not args.skip_roofline:
        print(f"Wrote {RESULTS_DIR / 'roofline_summary.json'}")
    return 0 if all(ok for _, ok in strong.values()) else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())