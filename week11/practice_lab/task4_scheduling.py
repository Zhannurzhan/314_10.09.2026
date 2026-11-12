"""Phase 4: skewed workload, static vs dynamic scheduling, load imbalance factor.

Run:   python task4_scheduling.py          (full run, about 3-5 minutes)
       python task4_scheduling.py quick    (N = 2,000, just to test it works)
"""
import math
import multiprocessing as mp
import sys
import time

from harness import BenchStats, emit_verify_tag
from task0_cal import STUDENT_ID          # your ID is set once, in task0_cal.py

P_PHYS = 8                       # i5-12450H: 4 P-cores + 4 E-cores
N_ITEMS = 8_000
CHUNKS = [1, 8, 64, 512]
WARMUP, RUNS = 1, 5              # fewer than harness default: each run takes seconds


def k_of(i: int) -> int:
    """Work for item i: 1 + floor((i/40)^2) heavy iterations (grows like i^2)."""
    return 1 + (i * i) // 1600                # exact integer form of floor((i/40)^2)


def do_item(i: int) -> float:
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    x = i * 1e-4
    for _ in range(k_of(i)):
        x = sqrt(abs(sin(x) + cos(x)) + 1.0)  # one heavy iteration: sin, cos, sqrt
    return x


# =================================================================== worker
def sched_worker(strategy, chunk, n, p, counter, t_worker, n_done,
                 wid, ts, runs, start, done):
    for _ in range(runs):
        start.wait()
        t0 = time.perf_counter_ns()
        ts[2 * wid] = t0
        count = 0
        if strategy == "block":                       # contiguous block static
            for i in range(wid * n // p, (wid + 1) * n // p):
                do_item(i)
                count += 1
        elif strategy == "cyclic":                    # round-robin static
            for i in range(wid, n, p):
                do_item(i)
                count += 1
        else:                                         # dynamic shared queue
            while True:
                with counter.get_lock():              # synchronized index counter
                    first = counter.value
                    counter.value = first + chunk
                if first >= n:
                    break
                for i in range(first, min(first + chunk, n)):
                    do_item(i)
                    count += 1
        t1 = time.perf_counter_ns()
        ts[2 * wid + 1] = t1
        t_worker[wid] = (t1 - t0) / 1e6               # this worker's active ms
        n_done[wid] = count
        done.wait()


# =================================================================== timing
def run_strategy(strategy, chunk, n, p):
    total = WARMUP + RUNS
    start, done = mp.Barrier(p + 1), mp.Barrier(p + 1)
    counter = mp.Value("i", 0)
    t_worker = mp.Array("d", p)
    n_done = mp.Array("i", p)
    ts = mp.Array("q", 2 * p, lock=False)
    procs = [mp.Process(target=sched_worker,
                        args=(strategy, chunk, n, p, counter, t_worker, n_done,
                              w, ts, total, start, done))
             for w in range(p)]
    for pr in procs:
        pr.start()
    runs = []                                         # (T_wall ms, lambda, per-worker ms)
    for _ in range(total):
        counter.value = 0                             # reset the queue for this run
        start.wait(timeout=1800)
        done.wait(timeout=1800)
        assert sum(n_done[:]) == n, "some items were skipped or done twice"
        tw = list(t_worker[:])
        t_wall = (max(ts[1::2]) - min(ts[0::2])) / 1e6
        lam = max(tw) / (sum(tw) / p)
        runs.append((t_wall, lam, tw))
    for pr in procs:
        pr.join()
    runs = runs[WARMUP:]
    walls = sorted(r[0] for r in runs)
    stats = BenchStats(walls[0], walls[RUNS // 2],
                       walls[(3 * RUNS) // 4] - walls[RUNS // 4])
    median_run = sorted(runs, key=lambda r: r[0])[RUNS // 2]
    return stats, median_run[1], median_run[2]


# =================================================================== main
if __name__ == "__main__":
    if "quick" in sys.argv:
        N_ITEMS = 2_000
        print("QUICK MODE: N = 2,000 (do NOT report these numbers)\n")

    p, n = P_PHYS, N_ITEMS
    total_work = sum(k_of(i) for i in range(n))
    last_block = sum(k_of(i) for i in range((p - 1) * n // p, n))

    plan = [("Block static", "block", 0), ("Cyclic static", "cyclic", 0)]
    plan += [(f"Dynamic C={c}", "dynamic", c) for c in CHUNKS]
    res = {}
    for k, (label, strat, c) in enumerate(plan, 1):
        print(f"[{k}/{len(plan)}] {label} ...", flush=True)
        res[label] = run_strategy(strat, c, n, p)

    dyn = [f"Dynamic C={c}" for c in CHUNKS]
    best = min(dyn, key=lambda lbl: res[lbl][0].median_ms)
    c_star = int(best.split("=")[1])
    t_block = res["Block static"][0].median_ms
    t_best = res[best][0].median_ms

    out = [f"### Phase 4: Skewed workload (N = {n:,}, p = {p})\n",
           f"Total work = {total_work:,} heavy iterations. "
           f"The last contiguous block (items {(p - 1) * n // p}-{n - 1}) holds "
           f"{last_block:,} = {100 * last_block / total_work:.1f}% of it "
           f"(a fair share is {100 / p:.1f}%).\n",
           "| Strategy | Min ms | Median T_wall ms | IQR ms | Speedup vs block | "
           "Imbalance lambda |",
           "|---|---:|---:|---:|---:|---:|"]
    for label, _, _ in plan:
        st, lam, _ = res[label]
        star = " **(C*)**" if label == best else ""
        out.append(f"| {label}{star} | {st.min_ms:.1f} | {st.median_ms:.1f} | "
                   f"{st.iqr_ms:.1f} | {t_block / st.median_ms:.2f}x | {lam:.3f} |")

    out += ["\n**Per-worker active time t_worker[t] in ms (median run)**\n",
            "| Worker | " + " | ".join(lbl for lbl in
                                     ["Block static", "Cyclic static", best]) + " |",
            "|---:|---:|---:|---:|"]
    for w in range(p):
        cells = [f"{res[lbl][2][w]:.1f}" for lbl in ["Block static", "Cyclic static", best]]
        out.append(f"| {w} | " + " | ".join(cells) + " |")

    out += [f"\nBest chunk size C* = {c_star}: T_wall = {t_best:.1f} ms, "
            f"{t_block / t_best:.2f}x faster than block static.\n"]
    out.append(emit_verify_tag("PHASE4", STUDENT_ID, t_block, t_best))

    text = "\n".join(out)
    print("\n" + text)
    with open("phase4_results.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nSaved: phase4_results.md")