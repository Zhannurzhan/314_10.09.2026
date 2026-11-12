"""Phase 1: false sharing stride sweep, process scaling and Karp-Flatt.

Run:   python task1_falsesharing.py          (full run, about 5 minutes)
       python task1_falsesharing.py quick    (10x smaller, just to test it works)
"""
import ctypes
import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory

from harness import BenchStats, emit_verify_tag
from task0_cal import STUDENT_ID          # your ID is set once, in task0_cal.py

P_PHYS = 8                       # physical cores: i5-12450H = 4 P-cores + 4 E-cores
M_TOTAL = 16_000_000             # total read-modify-write steps (1.6 x 10^7)
SHM_BYTES = 4096                 # 512 slots x 8 bytes
STRIDES = [1, 2, 4, 8, 16, 32]   # gap between workers' slots = 8 ... 256 bytes
WARMUP, RUNS = 2, 7              # same as harness.run_benchmark


# ---------------------------------------------------------------- worker
def fs_worker(shm_name: str, slot_idx: int, iters: int, start, done) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    start.wait()                              # all workers begin together
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc                   # store to the shared cache line
    done.wait()                               # report "finished" to the parent
    del arr
    shm.close()


# ---------------------------------------------------------------- timing
def one_run(shm_name: str, p: int, stride: int) -> float:
    """Start p workers, time ONLY the compute part (not process start-up)."""
    iters = M_TOTAL // p
    start = mp.Barrier(p + 1)
    done = mp.Barrier(p + 1)
    procs = [mp.Process(target=fs_worker,
                        args=(shm_name, w * stride, iters, start, done))
             for w in range(p)]
    for pr in procs:
        pr.start()
    start.wait(timeout=600)                   # wait until all workers are ready
    t0 = time.perf_counter_ns()
    done.wait(timeout=600)                    # wait until all workers finished
    t1 = time.perf_counter_ns()
    for pr in procs:
        pr.join()
    return (t1 - t0) / 1e6


def bench(shm_name: str, p: int, stride: int) -> BenchStats:
    for _ in range(WARMUP):
        one_run(shm_name, p, stride)
    s = sorted(one_run(shm_name, p, stride) for _ in range(RUNS))
    return BenchStats(s[0], s[RUNS // 2], s[(3 * RUNS) // 4] - s[RUNS // 4])


def mops(p: int, median_ms: float) -> float:
    return (M_TOTAL // p) * p / (median_ms * 1000.0)


def karp_flatt(speedup: float, p: int) -> float:
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    if "quick" in sys.argv:
        M_TOTAL //= 10
        print("QUICK MODE: M = 1.6e6 (do NOT report these numbers)\n")

    p_log = mp.cpu_count()
    p_list = sorted({2 ** k for k in range(10) if 2 ** k <= p_log} | {p_log})

    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    shm.buf[:] = bytes(SHM_BYTES)             # all slots start at 0
    results = {}                              # (p, stride) -> BenchStats

    def get(p, stride):
        if (p, stride) not in results:
            print(f"  running p={p:>2}, stride={stride:>2} ...", flush=True)
            results[(p, stride)] = bench(shm.name, p, stride)
        return results[(p, stride)]

    try:
        # ---- 1. stride sweep at p = P_phys
        print(f"[1/2] Stride sweep, p = P_phys = {P_PHYS}")
        for st in STRIDES:
            get(P_PHYS, st)

        # ---- 2. process scaling for stride 1 and 16
        print(f"[2/2] Process scaling, p in {p_list}")
        for st in (1, 16):
            for p in p_list:
                get(p, st)
    finally:
        shm.close()
        shm.unlink()

    # ---------------------------------------------------- build report text
    out = []
    out.append(f"### 1. Stride sweep (p = P_phys = {P_PHYS}, M = {M_TOTAL:,})\n")
    out.append("| Stride (slots) | Gap (bytes) | Min ms | Median ms | IQR ms | Mops/s |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for st in STRIDES:
        r = results[(P_PHYS, st)]
        out.append(f"| {st} | {st * 8} | {r.min_ms:.1f} | {r.median_ms:.1f} | "
                   f"{r.iqr_ms:.1f} | {mops(P_PHYS, r.median_ms):.2f} |")

    for st in (1, 16):
        t1 = results[(1, st)].median_ms
        out.append(f"\n### 2. Process scaling, stride = {st} ({st * 8} bytes)\n")
        out.append("| p | Median ms | IQR ms | Speedup S_p | Efficiency E_p | Karp-Flatt e(p) |")
        out.append("|---:|---:|---:|---:|---:|---:|")
        for p in p_list:
            r = results[(p, st)]
            sp = t1 / r.median_ms
            ep = sp / p
            kf = f"{karp_flatt(sp, p):.4f}" if p >= 2 else "-"
            out.append(f"| {p} | {r.median_ms:.1f} | {r.iqr_ms:.1f} | "
                       f"{sp:.3f} | {ep:.3f} | {kf} |")

    m1 = results[(P_PHYS, 1)].median_ms
    m2 = results[(P_PHYS, 16)].median_ms
    tag = emit_verify_tag("PHASE1", STUDENT_ID, m1, m2)
    out.append(f"\nstride=1 is {m1 / m2:.2f}x slower than stride=16 at p = {P_PHYS}.\n")
    out.append(tag)

    text = "\n".join(out)
    print("\n" + text)
    with open("phase1_results.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nSaved: phase1_results.md")