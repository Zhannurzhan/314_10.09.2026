"""Phase 3: empirical Roofline (bandwidth + compute ceilings) and cache-blocked GEMM.

Run:   python task3_roofline_gemm.py          (full run, about 1-3 minutes)
       python task3_roofline_gemm.py quick    (smaller sizes, just to test it works)
"""
import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory

from harness import BenchStats, emit_verify_tag
from task0_cal import STUDENT_ID          # your ID is set once, in task0_cal.py

P_PHYS = 8                       # i5-12450H: 4 P-cores + 4 E-cores
WARMUP, RUNS = 2, 7              # same as harness.run_benchmark

BW_BYTES = 64 * 10**6            # each copy buffer: 64 MB (formula uses 10^6)
FLOPS_PER_WORKER = 6 * 10**7     # compute ceiling: 6.0 x 10^7 FLOPs per worker
FLOPS_PER_ITER = 32              # 16 statements x (1 multiply + 1 add)
N = 256                          # GEMM size
BS_LIST = [8, 16, 32, 64, 96, 128]

# Your cache sizes (from Phase 0), used to say where each tile fits.
L1D_P, L1D_E = 48 * 1024, 32 * 1024
L2_P = 1280 * 1024               # 1.25 MB per P-core


# =================================================================== workers
def copy_worker(name_a, name_b, s, e, wid, ts, runs, start, done):
    sa = shared_memory.SharedMemory(name=name_a)
    sb = shared_memory.SharedMemory(name=name_b)
    for _ in range(runs):
        start.wait()
        ts[2 * wid] = time.perf_counter_ns()
        sa.buf[s:e] = sb.buf[s:e]            # memcpy: read B, write A
        ts[2 * wid + 1] = time.perf_counter_ns()
        done.wait()
    sa.close()
    sb.close()


def flop_worker(iters, wid, ts, runs, start, done):
    for _ in range(runs):
        x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
        x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
        m, c = 0.9999, 0.0001
        start.wait()
        ts[2 * wid] = time.perf_counter_ns()
        for _ in range(iters):                # 16 independent FMAs = 32 FLOPs
            x0 = x0 * m + c
            x1 = x1 * m + c
            x2 = x2 * m + c
            x3 = x3 * m + c
            x4 = x4 * m + c
            x5 = x5 * m + c
            x6 = x6 * m + c
            x7 = x7 * m + c
            x8 = x8 * m + c
            x9 = x9 * m + c
            x10 = x10 * m + c
            x11 = x11 * m + c
            x12 = x12 * m + c
            x13 = x13 * m + c
            x14 = x14 * m + c
            x15 = x15 * m + c
        ts[2 * wid + 1] = time.perf_counter_ns()
        done.wait()


def gemm_ijk(a, b, c, n, r0, r1, bs):
    """Variant 1: naive i-j-k. Inner k walks DOWN a column of B (stride n)."""
    for i in range(r0, r1):
        ia = i * n
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[ia + k] * b[k * n + j]
            c[ia + j] = s


def gemm_ikj(a, b, c, n, r0, r1, bs):
    """Variant 2: reordered i-k-j. a_ik hoisted, inner j is unit-stride."""
    for i in range(r0, r1):
        ia = i * n
        for k in range(n):
            aik = a[ia + k]
            kb = k * n
            for j in range(n):
                c[ia + j] += aik * b[kb + j]


def gemm_blocked(a, b, c, n, r0, r1, bs):
    """Variant 3: 6-loop cache-blocked, (i, k, j) micro-kernel, a_ik hoisted."""
    for ii in range(r0, r1, bs):
        i_end = min(ii + bs, r1)
        for kk in range(0, n, bs):
            k_end = min(kk + bs, n)
            for jj in range(0, n, bs):
                j_end = min(jj + bs, n)
                for i in range(ii, i_end):
                    ia = i * n
                    for k in range(kk, k_end):
                        aik = a[ia + k]
                        kb = k * n
                        for j in range(jj, j_end):
                            c[ia + j] += aik * b[kb + j]


KERNELS = {1: gemm_ijk, 2: gemm_ikj, 3: gemm_blocked}


def gemm_worker(variant, na, nb, nc, n, r0, r1, bs, wid, ts, runs, start, done):
    sa = shared_memory.SharedMemory(name=na)
    sb = shared_memory.SharedMemory(name=nb)
    sc = shared_memory.SharedMemory(name=nc)
    a, b, c = sa.buf.cast("d"), sb.buf.cast("d"), sc.buf.cast("d")
    kernel = KERNELS[variant]
    for _ in range(runs):
        start.wait()
        ts[2 * wid] = time.perf_counter_ns()
        for idx in range(r0 * n, r1 * n):     # clear my rows of C
            c[idx] = 0.0
        kernel(a, b, c, n, r0, r1, bs)
        ts[2 * wid + 1] = time.perf_counter_ns()
        done.wait()
    a.release(); b.release(); c.release()
    sa.close(); sb.close(); sc.close()


# =================================================================== timing
def run_parallel(target, per_worker_args):
    """Start workers once. Each run: time from the FIRST worker starting to the
    LAST worker finishing, using the workers' own timestamps (no start-up cost,
    no delay from waking the parent process)."""
    p = len(per_worker_args)
    total = WARMUP + RUNS
    start, done = mp.Barrier(p + 1), mp.Barrier(p + 1)
    ts = mp.Array("q", 2 * p, lock=False)    # [start_0, end_0, start_1, end_1, ...]
    procs = [mp.Process(target=target, args=(*args, w, ts, total, start, done))
             for w, args in enumerate(per_worker_args)]
    for pr in procs:
        pr.start()
    times = []
    for _ in range(total):
        start.wait(timeout=900)
        done.wait(timeout=900)
        span = max(ts[1::2]) - min(ts[0::2])
        times.append(span / 1e6)
    for pr in procs:
        pr.join()
    s = sorted(times[WARMUP:])
    return BenchStats(s[0], s[RUNS // 2], s[(3 * RUNS) // 4] - s[RUNS // 4])


def where_fits(nbytes):
    if nbytes <= L1D_E:
        return "L1d (all cores)"
    if nbytes <= L1D_P:
        return "L1d (P-core only)"
    if nbytes <= L2_P:
        return "L2"
    return "L3"


# =================================================================== main
if __name__ == "__main__":
    if "quick" in sys.argv:
        BW_BYTES //= 10
        FLOPS_PER_WORKER //= 10
        N = 128
        print("QUICK MODE: smaller sizes (do NOT report these numbers)\n")

    p = P_PHYS
    out = []

    # ------------------------------------------------ 3A.1 DRAM bandwidth
    print(f"[1/3] Memory bandwidth: 2 x {BW_BYTES // 10**6} MB, {p} workers", flush=True)
    sa = shared_memory.SharedMemory(create=True, size=BW_BYTES)
    sb = shared_memory.SharedMemory(create=True, size=BW_BYTES)
    sb.buf[:] = b"\x5a" * BW_BYTES
    try:
        bw_st = run_parallel(copy_worker,
                             [(sa.name, sb.name, w * BW_BYTES // p, (w + 1) * BW_BYTES // p)
                              for w in range(p)])
    finally:
        sa.close(); sa.unlink(); sb.close(); sb.unlink()
    bw = 2 * BW_BYTES / (bw_st.median_ms / 1000) / 1e9

    # ------------------------------------------------ 3A.2 compute ceiling
    print(f"[2/3] Compute ceiling: {FLOPS_PER_WORKER:.1e} FLOPs per worker, {p} workers",
          flush=True)
    iters = FLOPS_PER_WORKER // FLOPS_PER_ITER
    fl_st = run_parallel(flop_worker, [(iters,) for _ in range(p)])
    peak = p * iters * FLOPS_PER_ITER / (fl_st.median_ms / 1000) / 1e9
    ridge = peak / bw

    out += ["### Part 3A: Roofline ceilings (p = P_phys = 8)\n",
            "| Ceiling | Min ms | Median ms | IQR ms | Value |",
            "|---|---:|---:|---:|---:|",
            f"| Memory bandwidth BW_sustained | {bw_st.min_ms:.2f} | {bw_st.median_ms:.2f} | "
            f"{bw_st.iqr_ms:.2f} | {bw:.2f} GB/s |",
            f"| Compute P_sustained | {fl_st.min_ms:.1f} | {fl_st.median_ms:.1f} | "
            f"{fl_st.iqr_ms:.1f} | {peak:.4f} GFLOP/s |",
            f"\nRidge point I* = P_sustained / BW_sustained = {peak:.4f} / {bw:.2f} "
            f"= {ridge:.5f} FLOP/byte\n"]

    # ------------------------------------------------ 3B GEMM
    print(f"[3/3] GEMM N = {N}, {p} workers", flush=True)
    import random
    rnd = random.Random(42)
    nbytes = N * N * 8
    shm = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(3)]
    va, vb, vc = (s.buf.cast("d") for s in shm)
    for idx in range(N * N):
        va[idx] = rnd.uniform(-1, 1)
        vb[idx] = rnd.uniform(-1, 1)
    names = [s.name for s in shm]

    def gemm(variant, bs=0):
        args = [(variant, *names, N, w * N // p, (w + 1) * N // p, bs) for w in range(p)]
        st = run_parallel(gemm_worker, args)
        return st, list(vc)                   # timing + copy of C to check results

    flops = 2 * N ** 3
    gf = lambda st: flops / (st.median_ms / 1000) / 1e9
    rows = []                                 # (label, bs, stats, intensity)
    try:
        print("  Variant 1 (i-j-k) ...", flush=True)
        st1, c1 = gemm(1)
        rows.append(("V1 naive i-j-k", "-", st1, 2 / 16))
        print("  Variant 2 (i-k-j) ...", flush=True)
        st2, c2 = gemm(2)
        rows.append(("V2 reordered i-k-j", "-", st2, 2 / 24))
        err12 = max(abs(x - y) for x, y in zip(c1, c2))
        v3 = {}
        err3 = 0.0
        for bs in BS_LIST:
            print(f"  Variant 3 (blocked) BS = {bs} ...", flush=True)
            st3, c3 = gemm(3, bs)
            v3[bs] = st3
            err3 = max(err3, max(abs(x - y) for x, y in zip(c3, c2)))
            rows.append((f"V3 blocked", bs, st3, bs / 12))
    finally:
        va.release(); vb.release(); vc.release()
        for s in shm:
            s.close(); s.unlink()

    bs_star = min(v3, key=lambda b: v3[b].median_ms)
    out += [f"### Part 3B: GEMM (N = {N}, p = {p}, 2N^3 = {flops:,} FLOPs)\n",
            f"Correctness: max|C1 - C2| = {err12:.2e}, "
            f"max|C3 - C2| over all BS = {err3:.2e} "
            f"({'PASS' if err3 < 1e-9 and err12 < 1e-9 else 'FAIL'}, limit 1e-9)\n",
            "| Variant | BS | Tile footprint 3*BS^2*8 | Fits in | Median ms | IQR ms "
            "| GFLOP/s | Intensity I (FLOP/B) | Roofline bound GFLOP/s | % of bound |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for label, bs, st, inten in rows:
        if bs == "-":
            foot, fits = "-", "-"
        else:
            fb = 3 * bs * bs * 8
            foot, fits = f"{fb / 1024:.1f} KB", where_fits(fb)
        bound = min(peak, inten * bw)
        star = " **(BS*)**" if bs == bs_star else ""
        out.append(f"| {label}{star} | {bs} | {foot} | {fits} | {st.median_ms:.1f} | "
                   f"{st.iqr_ms:.1f} | {gf(st):.4f} | {inten:.3f} | {bound:.4f} | "
                   f"{100 * gf(st) / bound:.1f}% |")

    g1, g3 = gf(st1), gf(v3[bs_star])
    out += [f"\nBest tile BS* = {bs_star}: {g3:.4f} GFLOP/s "
            f"({g3 / g1:.2f}x Variant 1, {g3 / gf(st2):.2f}x Variant 2)",
            f"Every variant has I > I* = {ridge:.5f}, so all are on the COMPUTE roof.\n"]
    tag = emit_verify_tag("PHASE3", STUDENT_ID, g1, g3)
    out.append(tag)

    text = "\n".join(out)
    print("\n" + text)
    with open("phase3_results.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nSaved: phase3_results.md")