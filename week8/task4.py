"""
TASK 4 - Work Stealing & Scheduling Under Skew
"""

import platform
import queue
import threading
import time
from statistics import median

# ===================== CHANGE THIS =====================
STUDENT_ID = "230109009"  
# =======================================================

N_ITEMS = 1000
WORKERS = 4
REPEATS = 5                # each strategy runs 5 times, median reported
OUTPUT_FILE = "task4_output.txt"

lines = []


def log(text=""):
    """Print to the console AND remember the line for task4_output.txt."""
    print(text)
    lines.append(text)


# ---------------- Workloads ----------------
def cost_required(i):
    """Task spec: sleep(0.0001 * (i % 20)) -> 0 ms .. 1.9 ms, repeating pattern."""
    return 0.0001 * (i % 20)


def cost_growing(i):
    """Extra scenario: same 0 .. 1.9 ms range, but cost grows with i,
    so the LAST block of items is much heavier than the first."""
    return 0.0001 * (i // 50)


def do_item(i, cost_fn):
    """Simulate the work for one item."""
    t = cost_fn(i)
    if t > 0:
        time.sleep(t)


# ---------------- Strategies ----------------
def run_static(cost_fn):
    """Strategy A: split items into 4 equal blocks, one block per worker."""
    block = N_ITEMS // WORKERS
    ranges = [(w * block, N_ITEMS if w == WORKERS - 1 else (w + 1) * block)
              for w in range(WORKERS)]

    def job(w, finish, counts, start):
        lo, hi = ranges[w]
        for i in range(lo, hi):
            do_item(i, cost_fn)
        counts[w] = hi - lo
        finish[w] = time.perf_counter() - start

    return _launch(job)


def run_dynamic(cost_fn, chunk):
    """Strategy B: a shared queue of small chunks. A worker that finishes
    early simply grabs the next chunk, so nobody sits idle."""
    tasks = queue.SimpleQueue()
    for lo in range(0, N_ITEMS, chunk):
        tasks.put((lo, min(lo + chunk, N_ITEMS)))

    def job(w, finish, counts, start):
        done = 0
        while True:
            try:
                lo, hi = tasks.get_nowait()
            except queue.Empty:
                break
            for i in range(lo, hi):
                do_item(i, cost_fn)
            done += hi - lo
        counts[w] = done
        finish[w] = time.perf_counter() - start

    return _launch(job)


def _launch(job):
    """Start all workers together and collect total time + per-worker finish times."""
    finish = [0.0] * WORKERS
    counts = [0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)
    holder = {}

    def wrapper(w):
        barrier.wait()                       # everyone starts at the same moment
        job(w, finish, counts, holder["start"])

    threads = [threading.Thread(target=wrapper, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    holder["start"] = time.perf_counter()
    barrier.wait()
    for t in threads:
        t.join()
    total = time.perf_counter() - holder["start"]
    return total * 1000, [f * 1000 for f in finish], counts


def benchmark(name, fn):
    """Run one strategy REPEATS times; keep the run with the median total time."""
    runs = sorted((fn() for _ in range(REPEATS)), key=lambda r: r[0])
    total, finish, counts = runs[len(runs) // 2]
    first, last = min(finish), max(finish)
    return {"name": name, "total": total, "first": first, "last": last,
            "delta": last - first, "finish": finish, "counts": counts}


def run_scenario(title, cost_fn):
    work_ms = sum(cost_fn(i) for i in range(N_ITEMS)) * 1000
    log(f"Scenario: {title}")
    log(f"Total work = {work_ms:.1f} ms of sleep  ->  ideal time with "
        f"{WORKERS} workers = {work_ms / WORKERS:.1f} ms")
    log("-" * 78)

    results = [
        benchmark("A: Static blocks (250 each)", lambda: run_static(cost_fn)),
        benchmark("B: Dynamic, chunk = 1", lambda: run_dynamic(cost_fn, 1)),
        benchmark("B: Dynamic, chunk = 5", lambda: run_dynamic(cost_fn, 5)),
    ]

    log(f"{'Strategy':<28} | {'Total ms':>9} | {'First done':>10} | "
        f"{'Last done':>9} | {'Delta ms':>8}")
    log("-" * 78)
    for r in results:
        log(f"{r['name']:<28} | {r['total']:>9.1f} | {r['first']:>10.1f} | "
            f"{r['last']:>9.1f} | {r['delta']:>8.1f}")
    log("-" * 78)
    for r in results:
        per_worker = ", ".join(f"W{w}:{c} items @ {f:.0f} ms"
                               for w, (c, f) in enumerate(zip(r["counts"], r["finish"])))
        log(f"  {r['name']:<26} -> {per_worker}")
    static_t = results[0]["total"]
    best = min(results[1:], key=lambda r: r["total"])
    log(f"  Static / best dynamic ({best['name'][3:]}) = {static_t / best['total']:.2f}x")
    log("=" * 78)


def main():
    log("=" * 78)
    log("TASK 4 - Work Stealing & Scheduling Under Skew")
    log("=" * 78)
    log(f"Student ID       : {STUDENT_ID}")
    log(f"OS               : {platform.platform()}")
    log(f"Python version   : {platform.python_version()}")
    log(f"Items            : {N_ITEMS} (indices 0..{N_ITEMS - 1})")
    log(f"Workers          : {WORKERS} threads (sleep releases the GIL)")
    log(f"Repeats          : {REPEATS} per strategy (median run reported)")
    log(f"Delta            : time between FIRST and LAST worker finishing")
    log("=" * 78)

    run_scenario("REQUIRED  cost(i) = sleep(0.0001 * (i % 20))", cost_required)
    run_scenario("EXTRA     cost(i) = sleep(0.0001 * (i // 50))  (cost grows with i)",
                 cost_growing)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(Output also saved to {OUTPUT_FILE})")


if __name__ == "__main__":
    main()