"""Phase 2: strong scaling (Amdahl) vs weak scaling (Gustafson).

Run:   python task2_scaling.py          (full run, about 4-6 minutes)
       python task2_scaling.py quick    (10x smaller, just to test it works)
"""
import math
import multiprocessing as mp
import sys
import time

from harness import BenchStats, emit_verify_tag
from task0_cal import STUDENT_ID          # your ID is set once, in task0_cal.py

P_PHYS = 8                 # physical cores: i5-12450H = 4 P-cores + 4 E-cores
N_STRONG = 1_200_000       # strong scaling: total elements stay fixed
N0_WEAK = 200_000          # weak scaling: elements PER WORKER stay fixed
SERIAL_SHARE = 0.05        # Stage A covers the first 5% of the elements
WARMUP, RUNS = 2, 7        # same as harness.run_benchmark


# ---------------------------------------------------------------- Stage A (serial)
def stage_a(n_a: int) -> float:
    """Runs in the parent only. 10 math ops per element (5 x sin + 5 x sqrt)."""
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(n_a):
        x = i * 1e-6
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        acc += x
    return acc


# ---------------------------------------------------------------- Stage B (parallel)
def stage_b_chunk(args) -> float:
    """Runs in a pool worker. 30 ops per element: 10 x (FMA, sin, FMA).
    Only returns a number, never writes to shared memory."""
    start, end, seed = args
    sin = math.sin
    acc = 0.0
    for i in range(start, end):
        x = i * 1e-6 + seed
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        x = sin(x * 0.999 + 0.001) * 1.001 + 0.0005
        acc += x
    return acc


# ---------------------------------------------------------------- one trial
def one_trial(pool, p: int, n: int):
    """Stage A in the parent, then Stage B split into p equal blocks."""
    t0 = time.perf_counter_ns()
    seed = stage_a(int(SERIAL_SHARE * n)) * 1e-9      # B depends on A's result
    t1 = time.perf_counter_ns()
    bounds = [(w * n // p, (w + 1) * n // p, seed) for w in range(p)]
    sum(pool.map(stage_b_chunk, bounds, chunksize=1))
    t2 = time.perf_counter_ns()
    return (t2 - t0) / 1e6, (t1 - t0) / 1e6           # total ms, Stage A ms


def bench(p: int, n: int):
    """Persistent pool made OUTSIDE the timed loop. Returns (stats, Stage A median)."""
    with mp.Pool(processes=p) as pool:
        for _ in range(WARMUP):
            one_trial(pool, p, n)
        trials = [one_trial(pool, p, n) for _ in range(RUNS)]
    tot = sorted(t for t, _ in trials)
    a = sorted(a for _, a in trials)
    stats = BenchStats(tot[0], tot[RUNS // 2], tot[(3 * RUNS) // 4] - tot[RUNS // 4])
    return stats, a[RUNS // 2]


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    if "quick" in sys.argv:
        N_STRONG //= 10
        N0_WEAK //= 10
        print("QUICK MODE: N is 10x smaller (do NOT report these numbers)\n")

    p_log = mp.cpu_count()
    ps = list(range(1, p_log + 1))

    # ---- 1. strong scaling
    print(f"[1/2] Strong scaling, N = {N_STRONG:,}")
    strong = {}
    for p in ps:
        print(f"  p = {p:>2} ...", flush=True)
        strong[p] = bench(p, N_STRONG)

    # ---- 2. weak scaling
    print(f"[2/2] Weak scaling, N(p) = p x {N0_WEAK:,}")
    weak = {}
    for p in ps:
        print(f"  p = {p:>2} ...", flush=True)
        weak[p] = bench(p, p * N0_WEAK)

    # ---------------------------------------------------- strong table
    t1s = strong[1][0].median_ms
    s_frac = strong[1][1] / t1s                     # Amdahl s = T_A(1) / T_1
    cap = 1.0 / s_frac
    out = [f"### 1. Strong scaling (N = {N_STRONG:,})\n",
           f"Amdahl serial fraction s = T_StageA(1) / T_1 = "
           f"{strong[1][1]:.1f} / {t1s:.1f} = {s_frac:.5f}",
           f"Amdahl cap S_inf = 1/s = {cap:.1f}\n",
           "| p | Stage A ms | Median ms | IQR ms | Speedup S_p | Efficiency E_p "
           "| Amdahl predicts | Marginal dS |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sp = {}
    for p in ps:
        st, ta = strong[p]
        sp[p] = t1s / st.median_ms
        amd = 1.0 / (s_frac + (1.0 - s_frac) / p)
        d = f"{sp[p] - sp[p - 1]:+.3f}" if p > 1 else "-"
        out.append(f"| {p} | {ta:.1f} | {st.median_ms:.1f} | {st.iqr_ms:.1f} | "
                   f"{sp[p]:.3f} | {sp[p] / p:.3f} | {amd:.3f} | {d} |")
    out.append(f"\nObserved S at P_log={p_log}: {sp[p_log]:.3f}  "
               f"({100 * sp[p_log] / cap:.1f}% of the Amdahl cap {cap:.1f})")

    # ---------------------------------------------------- inflection
    if p_log > P_PHYS:
        d_before = sp[P_PHYS] - sp[P_PHYS - 1]
        d_after = sp[P_PHYS + 1] - sp[P_PHYS]
        out.append(f"Marginal speedup {P_PHYS - 1}->{P_PHYS}: {d_before:+.3f}, "
                   f"{P_PHYS}->{P_PHYS + 1}: {d_after:+.3f}  "
                   f"(change: {d_after - d_before:+.3f})")

    # ---------------------------------------------------- weak table
    t1w = weak[1][0].median_ms
    out += [f"\n### 2. Weak scaling (N0 = {N0_WEAK:,} per worker)\n",
            "| p | N(p) | Stage A ms | Median ms | IQR ms | E_weak = T1/Tp "
            "| alpha = T_A/T_p | S_Gustafson | Measured scaled S = p*E_weak |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sg = {}
    for p in ps:
        st, ta = weak[p]
        e_w = t1w / st.median_ms
        alpha = ta / st.median_ms
        sg[p] = p - alpha * (p - 1)
        out.append(f"| {p} | {p * N0_WEAK:,} | {ta:.1f} | {st.median_ms:.1f} | "
                   f"{st.iqr_ms:.1f} | {e_w:.3f} | {alpha:.4f} | {sg[p]:.3f} | "
                   f"{p * e_w:.3f} |")

    tag = emit_verify_tag("PHASE2", STUDENT_ID, sp[p_log], sg[p_log])
    out.append(f"\nS_strong(P_log) = {sp[p_log]:.4f}, "
               f"S_Gustafson(P_log) = {sg[p_log]:.4f}\n")
    out.append(tag)

    text = "\n".join(out)
    print("\n" + text)
    with open("phase2_results.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nSaved: phase2_results.md")