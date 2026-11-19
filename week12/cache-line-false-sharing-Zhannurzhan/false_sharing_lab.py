import os
import re
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
from datetime import datetime, timezone
import numpy as np
import numba
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "Zhannurzhan"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

# ---- Additional lab constants --------------------------------------------------
STRIDES = (1, 4, 8, 16)
LAYOUT_NAMES = {1: "A_contiguous", 4: "B_half_line", 8: "C_line_padded", 16: "D_prefetch_safe"}
REPEATS = 3                       # each layout is run 3x; the median is reported
BUFFER_ALIGN = 128                # base aligned to 128 B (=> also 64 B aligned)
BUFFER_ELEMS = 16 * NUM_THREADS + 16
WARMUP_ITERS = 10_000
RUN_GIL_CONTRAST = True           # small pure-Python experiment for Q5
GIL_CONTRAST_ITERS = 1_000_000
MANIFEST_PATH = "cache_lab_manifest.json"
LLVM_DUMP_PATH = "worker_strided_accumulate.ll"
GOLDEN64 = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def _linux_core_key(cpu: int):
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    try:
        with open(f"{base}/core_id", encoding="utf-8") as f:
            core = int(f.read().strip())
        with open(f"{base}/physical_package_id", encoding="utf-8") as f:
            pkg = int(f.read().strip())
        return (pkg, core)
    except (OSError, ValueError):
        return (0, cpu)


def choose_pinned_cpus(n: int):
    """Pick one logical CPU per physical core first (avoid SMT siblings sharing an L1)."""
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        avail = sorted(os.sched_getaffinity(0))
        seen, primary, siblings = set(), [], []
        for c in avail:
            key = _linux_core_key(c)
            if key in seen:
                siblings.append(c)
            else:
                seen.add(key)
                primary.append(c)
        return (primary + siblings)[:n], len(seen), "linux_sched_setaffinity"
    if sys.platform == "win32":
        return list(range(n)), None, "win32_SetThreadAffinityMask"
    return [], None, "unsupported (OS has no thread-affinity API, e.g. macOS)"


def pin_current_thread(cpu, method) -> bool:
    try:
        if method == "linux_sched_setaffinity":
            os.sched_setaffinity(0, {cpu})          # pid 0 = calling thread on Linux
            return True
        if method == "win32_SetThreadAffinityMask":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            return k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << cpu) != 0
    except Exception:
        return False
    return False


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # Task 1: deterministic 64-bit seed = first 16 hex digits of SHA-256(git_user::cpu_sig)
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)
    thread_seeds = [(instance_seed + (t + 1) * GOLDEN64) & MASK64 for t in range(NUM_THREADS)]

    # Affinity masks (when supported by the OS)
    pinned_cpus, physical_cores, affinity_method = choose_pinned_cpus(NUM_THREADS)

    return {
        "git_username": git_user.strip(),
        "cpu_sig": cpu_sig,
        "cpu_model": read_cpu_model(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "cache_line_size": line_size,
        "instance_seed": instance_seed,
        "thread_seeds": thread_seeds,
        "pinned_cpus": pinned_cpus,
        "affinity_method": affinity_method,
        "affinity_supported": affinity_method in ("linux_sched_setaffinity", "win32_SetThreadAffinityMask"),
    }


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    top = np.uint64(63)
    for k in range(iters):
        v = buffer[slot]                                   # load the (shared) cache line
        bit = (seed >> np.uint64(k & 63)) & one            # seed-dependent increment: 1 or 2
        # Counters never exceed 2**63, so (v >> 63) == 0 at runtime and idx == slot.
        # LLVM cannot prove that, so the store address depends on the value loaded in
        # this iteration -> not loop-invariant -> no register promotion / no hoisting.
        idx = slot + np.int64(v >> top)
        buffer[idx] = v + bit + one                        # real i64 store every iteration
    return buffer[slot]


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    one = np.uint64(1)
    acc = buffer[slot]                                     # one load
    for k in range(iters):
        acc += ((seed >> np.uint64(k & 63)) & one) + one   # register-only accumulation
    buffer[slot] = acc                                     # exactly one store
    return acc


def python_strided_accumulate(buffer, slot, iters, seed):
    """Pure-Python (GIL-bound) version of the same update, used only for Q5."""
    for k in range(iters):
        v = buffer[slot]
        buffer[slot] = v + ((seed >> (k & 63)) & 1) + 1
    return buffer[slot]


def expected_counter(seed: int, iters: int) -> int:
    """Closed-form final counter value: each 64-iteration block adds popcount(seed) + 64."""
    full, rem = divmod(iters, 64)
    tail = sum(((seed >> b) & 1) + 1 for b in range(rem))
    return full * (bin(seed).count("1") + 64) + tail


def aligned_zeros_uint64(n_elems: int, align: int):
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    buf = raw[offset: offset + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % align == 0 and buf.flags.c_contiguous
    return buf


def _rusage():
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (int(r.ru_nvcsw), int(r.ru_nivcsw))


def _run_threaded(kernel, buffer, slots, iters, seeds, ctx):
    """Start len(slots) threads, pin them, release them together, time until all join."""
    n = len(slots)
    barrier = threading.Barrier(n + 1)
    out = [None] * n
    errors = []
    cpus, method = ctx["pinned_cpus"], ctx["affinity_method"]

    def body(t):
        cpu = cpus[t] if t < len(cpus) else None
        pinned = pin_current_thread(cpu, method) if cpu is not None else False
        barrier.wait()
        try:
            t0 = time.perf_counter_ns()
            val = kernel(buffer, slots[t], iters, seeds[t])
            t1 = time.perf_counter_ns()
            out[t] = {"thread": t, "cpu": cpu, "pinned": pinned, "elapsed_ns": t1 - t0,
                      "value": int(val), "native_tid": threading.get_native_id()}
        except Exception as exc:  # surface errors to the main thread
            errors.append(repr(exc))

    threads = [threading.Thread(target=body, args=(t,), name=f"fs-worker-{t}") for t in range(n)]
    ru0 = _rusage()
    for th in threads:
        th.start()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    t1 = time.perf_counter_ns()
    ru1 = _rusage()
    if errors:
        raise RuntimeError(f"worker thread failed: {errors}")
    return {
        "wall_ns": t1 - t0,
        "threads": out,
        "voluntary_ctx_switches": None if ru0 is None else ru1[0] - ru0[0],
        "involuntary_ctx_switches": None if ru0 is None else ru1[1] - ru0[1],
    }


def _summarize_layout(name, stride, slots, runs, ctx):
    walls = [r["wall_ns"] for r in runs]
    med_idx = int(np.argsort(walls)[len(walls) // 2])
    med = runs[med_idx]
    base = runs[med_idx]["base_address"]
    line = ctx["cache_line_size"]
    offsets = [8 * s for s in slots]
    line_idx = [((base + o) // line) - (base // line) for o in offsets]
    sector128 = [((base + o) // 128) - (base // 128) for o in offsets]
    wall = med["wall_ns"]
    return {
        "layout": name,
        "stride": stride,
        "byte_spacing": 8 * stride,
        "slots": slots,
        "byte_offsets": offsets,
        "cache_line_index_relative": line_idx,
        "distinct_cache_lines": len(set(line_idx)),
        "distinct_128B_sectors": len(set(sector128)),
        "base_address_mod_64": base % 64,
        "base_address_mod_128": base % 128,
        "wall_ns_runs": walls,
        "wall_ns_median": wall,
        "ns_per_update_per_thread": wall / ITERATIONS_PER_THREAD,
        "ns_per_update_aggregate": wall / (len(slots) * ITERATIONS_PER_THREAD),
        "per_thread_ns": [t["elapsed_ns"] for t in med["threads"]],
        "pinned_cpus": [t["cpu"] for t in med["threads"]],
        "pinning_ok": all(t["pinned"] for t in med["threads"]),
        "voluntary_ctx_switches": med["voluntary_ctx_switches"],
        "involuntary_ctx_switches": med["involuntary_ctx_switches"],
        "final_values": med["final_values"],
        "final_values_all_runs": [r["final_values"] for r in runs],
    }


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx) -> list:
    seeds = [np.uint64(s) for s in ctx["thread_seeds"]]
    results = []
    for stride in STRIDES:
        slots = [t * stride for t in range(NUM_THREADS)]
        runs = []
        for _ in range(REPEATS):
            buf = aligned_zeros_uint64(BUFFER_ELEMS, BUFFER_ALIGN)
            r = _run_threaded(worker_strided_accumulate, buf, slots, ITERATIONS_PER_THREAD, seeds, ctx)
            r["final_values"] = [int(buf[s]) for s in slots]
            r["base_address"] = int(buf.ctypes.data)
            runs.append(r)
        res = _summarize_layout(LAYOUT_NAMES[stride], stride, slots, runs, ctx)
        results.append(res)
        print(f"  stride={stride:>2} ({res['byte_spacing']:>3} B)  median={res['wall_ns_median']:>14,} ns"
              f"  ({res['ns_per_update_per_thread']:.3f} ns/update/thread)"
              f"  lines={res['distinct_cache_lines']}  runs={res['wall_ns_runs']}", flush=True)
    return results


def run_private_baseline(ctx) -> dict:
    seeds = [np.uint64(s) for s in ctx["thread_seeds"]]
    slots = [t for t in range(NUM_THREADS)]          # contiguous slots, single final store each
    runs = []
    for _ in range(REPEATS):
        buf = aligned_zeros_uint64(BUFFER_ELEMS, BUFFER_ALIGN)
        r = _run_threaded(worker_private_accumulate, buf, slots, ITERATIONS_PER_THREAD, seeds, ctx)
        r["final_values"] = [int(buf[s]) for s in slots]
        r["base_address"] = int(buf.ctypes.data)
        runs.append(r)
    res = _summarize_layout("private_register", 1, slots, runs, ctx)
    print(f"  private (register)  median={res['wall_ns_median']:>14,} ns"
          f"  ({res['ns_per_update_per_thread']:.3f} ns/update/thread)  runs={res['wall_ns_runs']}", flush=True)
    return res


def run_gil_contrast(ctx) -> dict:
    """Same stride=1 update with pure-Python threads vs nogil Numba threads, 1 vs 4 threads."""
    iters = GIL_CONTRAST_ITERS
    out = {"iters_per_thread": iters, "switch_interval_s": sys.getswitchinterval(),
           "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)())}
    for n in (1, NUM_THREADS):
        slots = list(range(n))
        pbuf = [0] * BUFFER_ELEMS
        rp = _run_threaded(python_strided_accumulate, pbuf, slots, iters, ctx["thread_seeds"][:n], ctx)
        nbuf = aligned_zeros_uint64(BUFFER_ELEMS, BUFFER_ALIGN)
        rn = _run_threaded(worker_strided_accumulate, nbuf, slots, iters,
                           [np.uint64(s) for s in ctx["thread_seeds"][:n]], ctx)
        ok = all(pbuf[s] == expected_counter(ctx["thread_seeds"][s], iters) for s in slots)
        out[f"python_threads_{n}"] = {"wall_ns": rp["wall_ns"],
                                      "voluntary_ctx_switches": rp["voluntary_ctx_switches"],
                                      "involuntary_ctx_switches": rp["involuntary_ctx_switches"],
                                      "values_correct": ok}
        out[f"numba_nogil_threads_{n}"] = {"wall_ns": rn["wall_ns"]}
    out["python_4_over_1"] = out[f"python_threads_{NUM_THREADS}"]["wall_ns"] / out["python_threads_1"]["wall_ns"]
    out["numba_4_over_1"] = out[f"numba_nogil_threads_{NUM_THREADS}"]["wall_ns"] / out["numba_nogil_threads_1"]["wall_ns"]
    print(f"  pure Python : 1 thread {out['python_threads_1']['wall_ns']:,} ns | "
          f"{NUM_THREADS} threads {out[f'python_threads_{NUM_THREADS}']['wall_ns']:,} ns  "
          f"(x{out['python_4_over_1']:.3f})")
    print(f"  Numba nogil : 1 thread {out['numba_nogil_threads_1']['wall_ns']:,} ns | "
          f"{NUM_THREADS} threads {out[f'numba_nogil_threads_{NUM_THREADS}']['wall_ns']:,} ns  "
          f"(x{out['numba_4_over_1']:.3f})", flush=True)
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def _split_llvm_functions(ir: str) -> dict:
    funcs, cur, buf = {}, None, []
    for line in ir.splitlines():
        if line.startswith("define "):
            cur, buf = line, [line]
        elif cur is not None:
            buf.append(line)
            if line.startswith("}"):
                funcs[cur] = "\n".join(buf)
                cur = None
    return funcs


def _native_function(dispatcher, name):
    llvm = dispatcher.inspect_llvm()
    ir = "\n".join(llvm.values()) if isinstance(llvm, dict) else llvm
    for header, body in _split_llvm_functions(ir).items():
        if name in header and "cpython" not in header and "cfunc" not in header:
            return header, body, ir
    return None, "", ir


def _loop_store_analysis(fn_ir: str) -> dict:
    """Find basic blocks inside loops (back-edges) and the i64 stores/loads they contain."""
    blocks, label, lines = [], "entry", []
    for line in fn_ir.splitlines()[1:]:
        m = re.match(r"^([A-Za-z0-9_.$\-]+):", line)
        if m:
            blocks.append((label, lines))
            label, lines = m.group(1), []
        else:
            lines.append(line)
    blocks.append((label, lines))
    index = {lab: i for i, (lab, _) in enumerate(blocks)}
    in_loop = set()
    for i, (_, blines) in enumerate(blocks):
        for line in blines:
            if line.strip().startswith("br "):
                for tgt in re.findall(r"label %([A-Za-z0-9_.$\-]+)", line):
                    j = index.get(tgt)
                    if j is not None and j <= i:
                        in_loop.update(range(j, i + 1))
    loop_stores = [l.strip() for i, (_, bl) in enumerate(blocks) if i in in_loop
                   for l in bl if re.search(r"\bstore (volatile )?i64\b", l)]
    loop_loads = [l.strip() for i, (_, bl) in enumerate(blocks) if i in in_loop
                  for l in bl if re.search(r"= load (volatile )?i64\b", l)]
    return {
        "basic_blocks": len(blocks),
        "loop_blocks": [blocks[i][0] for i in sorted(in_loop)],
        "i64_stores_total": len(re.findall(r"\bstore (?:volatile )?i64\b", fn_ir)),
        "i64_loads_total": len(re.findall(r"= load (?:volatile )?i64\b", fn_ir)),
        "i64_stores_in_loop": len(loop_stores),
        "i64_loads_in_loop": len(loop_loads),
        "loop_store_instructions": loop_stores[:8],
        "loop_load_instructions": loop_loads[:8],
        "volatile_ops": len(re.findall(r"\bvolatile\b", fn_ir)),
        "python_object_calls": len(re.findall(r"@Py|PyObject", fn_ir)),
    }


def audit_llvm_and_parity(ctx, layouts, private, gil) -> dict:
    # ---- Parity: every layout / every repeat / private baseline vs closed form ----
    expected = [expected_counter(s, ITERATIONS_PER_THREAD) for s in ctx["thread_seeds"]]
    parity_rows = {}
    for res in layouts + [private]:
        parity_rows[res["layout"]] = all(v == expected for v in res["final_values_all_runs"])
    parity_passed = all(parity_rows.values())

    # ---- LLVM IR audit ------------------------------------------------------------
    s_hdr, s_body, s_full = _native_function(worker_strided_accumulate, "worker_strided_accumulate")
    p_hdr, p_body, _ = _native_function(worker_private_accumulate, "worker_private_accumulate")
    s_an = _loop_store_analysis(s_body) if s_body else {}
    p_an = _loop_store_analysis(p_body) if p_body else {}
    nopython = len(worker_strided_accumulate.nopython_signatures) > 0
    llvm_passed = bool(s_body) and nopython and s_an.get("i64_stores_in_loop", 0) > 0 \
        and s_an.get("python_object_calls", 1) == 0
    with open(LLVM_DUMP_PATH, "w", encoding="utf-8") as f:
        f.write(s_full)

    # ---- Ratios -------------------------------------------------------------------
    by = {r["stride"]: r for r in layouts}
    ns = {s: by[s]["wall_ns_median"] for s in STRIDES}
    p_ns = private["wall_ns_median"]
    ratios = {
        "stride1_over_stride8": ns[1] / ns[8],
        "stride4_over_stride8": ns[4] / ns[8],
        "stride16_over_stride8": ns[16] / ns[8],
        "stride1_over_private": ns[1] / p_ns,
        "stride8_over_private": ns[8] / p_ns,
    }

    verification_passed = bool(parity_passed and llvm_passed and ctx["cache_line_size"] > 0)

    manifest = {
        "schema_version": "1.0",
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
        "thread_seeds": ctx["thread_seeds"],
        "cpu_sig": ctx["cpu_sig"],
        "cpu_model": ctx["cpu_model"],
        "os": ctx["os"],
        "logical_cores": ctx["logical_cores"],
        "physical_cores": ctx["physical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "affinity_supported": ctx["affinity_supported"],
        "affinity_method": ctx["affinity_method"],
        "pinned_cpus": ctx["pinned_cpus"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "buffer_alignment_bytes": BUFFER_ALIGN,
        "layouts": layouts,
        "private_baseline": private,
        "slowdown_ratios": ratios,
        "expected_final_values": expected,
        "parity_by_layout": parity_rows,
        "parity_passed": parity_passed,
        "llvm_audit": {
            "nopython_signatures": [str(s) for s in worker_strided_accumulate.nopython_signatures],
            "strided_native_function": s_hdr,
            "strided": s_an,
            "private_native_function": p_hdr,
            "private": p_an,
            "llvm_dump_file": LLVM_DUMP_PATH,
            "passed": llvm_passed,
        },
        "gil_contrast": gil,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verification_passed": verification_passed,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    git_user = sys.argv[1] if len(sys.argv) > 1 else GIT_USERNAME
    ctx = init_system_context(git_user)
    print("=" * 78)
    print("Cache Line & False Sharing Mechanics")
    print("=" * 78)
    print(f"GIT_USERNAME      : {ctx['git_username']}")
    print(f"CPU model         : {ctx['cpu_model']}")
    print(f"cpu_sig           : {ctx['cpu_sig']}")
    print(f"OS                : {ctx['os']}")
    print(f"Logical / physical: {ctx['logical_cores']} / {ctx['physical_cores']}")
    print(f"Cache line size   : {ctx['cache_line_size']} B")
    print(f"INSTANCE_SEED     : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    print(f"Affinity          : {ctx['affinity_method']}  pinned CPUs = {ctx['pinned_cpus']}")
    print(f"Threads x iters   : {NUM_THREADS} x {ITERATIONS_PER_THREAD:,}  (repeats={REPEATS}, median reported)")
    print("-" * 78, flush=True)

    # JIT warm-up (compilation is excluded from all timings)
    wb = aligned_zeros_uint64(BUFFER_ELEMS, BUFFER_ALIGN)
    worker_strided_accumulate(wb, 0, WARMUP_ITERS, np.uint64(ctx["thread_seeds"][0]))
    worker_private_accumulate(wb, 1, WARMUP_ITERS, np.uint64(ctx["thread_seeds"][0]))

    print("Task 3 - stride sweep (nogil Numba threads):")
    layouts = run_stride_benchmark(ctx)
    print("Task 4 - register-privatized baseline:")
    private = run_private_baseline(ctx)
    gil = None
    if RUN_GIL_CONTRAST:
        print(f"Q5 support - GIL contrast ({GIL_CONTRAST_ITERS:,} updates/thread, stride=1):")
        gil = run_gil_contrast(ctx)
    print("-" * 78)

    m = audit_llvm_and_parity(ctx, layouts, private, gil)
    r = m["slowdown_ratios"]
    a = m["llvm_audit"]["strided"]
    pa = m["llvm_audit"]["private"]
    print("Summary (median wall ns, 4 threads concurrently)")
    print(f"  {'layout':<18}{'stride':>7}{'wall_ns':>16}{'ns/update/thr':>15}{'lines':>7}{'vcsw':>7}{'ivcsw':>7}")
    for res in m["layouts"] + [m["private_baseline"]]:
        print(f"  {res['layout']:<18}{res['stride']:>7}{res['wall_ns_median']:>16,}"
              f"{res['ns_per_update_per_thread']:>15.3f}{res['distinct_cache_lines']:>7}"
              f"{str(res['voluntary_ctx_switches']):>7}{str(res['involuntary_ctx_switches']):>7}")
    print(f"  slowdown stride1/stride8   : {r['stride1_over_stride8']:.3f}x")
    print(f"  slowdown stride4/stride8   : {r['stride4_over_stride8']:.3f}x")
    print(f"  ratio    stride16/stride8  : {r['stride16_over_stride8']:.3f}x")
    print(f"  stride1 / private          : {r['stride1_over_private']:.3f}x")
    print(f"  stride8 / private          : {r['stride8_over_private']:.3f}x")
    print("-" * 78)
    print("LLVM IR audit (worker_strided_accumulate.inspect_llvm())")
    print(f"  native function  : {m['llvm_audit']['strided_native_function']}")
    print(f"  nopython sigs    : {m['llvm_audit']['nopython_signatures']}")
    print(f"  i64 stores total / in loop : {a.get('i64_stores_total')} / {a.get('i64_stores_in_loop')}")
    print(f"  i64 loads  total / in loop : {a.get('i64_loads_total')} / {a.get('i64_loads_in_loop')}")
    for line in a.get("loop_store_instructions", []):
        print(f"    loop store: {line}")
    print(f"  Python object calls in native fn: {a.get('python_object_calls')}")
    print(f"  private kernel i64 stores total / in loop: {pa.get('i64_stores_total')} / {pa.get('i64_stores_in_loop')}")
    print(f"  llvm_audit_passed: {m['llvm_audit']['passed']}")
    print("-" * 78)
    print(f"Expected final counters : {m['expected_final_values']}")
    print(f"Parity by layout        : {m['parity_by_layout']}")
    print(f"parity_passed           : {m['parity_passed']}")
    print(f"verification_passed     : {m['verification_passed']}")
    print(f"{MANIFEST_PATH} written; LLVM IR saved to {LLVM_DUMP_PATH}")

    assert m["parity_passed"], "Bit-exact parity failed across layouts"
    assert m["verification_passed"], "Verification failed (see manifest)"


if __name__ == "__main__":
    main()