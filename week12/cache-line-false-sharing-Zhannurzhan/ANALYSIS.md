# Cache Line & False Sharing Mechanics: Analysis

**Run identity (from `cache_lab_manifest.json`)**

| Field | Value |
|---|---|
| git_username | `Zhannurzhan` |
| CPU model | `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel` |
| cpu_sig | `AMD64::Intel64 Family 6 Model 154 Stepping 3, GenuineIntel::12` |
| OS | Windows 11 |
| Logical / physical cores | 12 / null (physical-core topology is not exposed on Windows) |
| Cache line size | 64 B |
| INSTANCE_SEED | 7101537537890963548 (`0x628dbac3e117d05c`) |
| Thread seeds | 55508283504610417, 11456223102827808902, 4410193848441455771, 15810908667764654256 |
| Affinity | `win32_SetThreadAffinityMask`, pinned CPUs [0, 1, 2, 3], `pinning_ok = true` |
| Workload | 4 threads × 60,000,000 updates, 3 repeats, median reported |
| Buffer alignment | 128 B (`base_address_mod_64 = 0`, `base_address_mod_128 = 0`) |
| Versions | Python 3.14.0, NumPy 2.3.5, Numba 0.63.1 |
| Expected final counters | [83437500, 94687500, 92812500, 89062500] |
| parity_passed | true |
| llvm_audit passed | true |
| **verification_passed** | **true** |

| Layout | Stride | Byte offsets | Line index (relative) | Distinct lines | Distinct 128 B sectors | Runs (ns) | Median wall (ns) | ns / update / thread | ns / update (aggregate) |
|---|---|---|---|---|---|---|---|---|---|
| A_contiguous | 1 | 0, 8, 16, 24 | 0, 0, 0, 0 | 1 | 1 | 1164434600, 1066861200, 1118423200 | **1,118,423,200** | 18.640 | 4.660096666666667 |
| B_half_line | 4 | 0, 32, 64, 96 | 0, 0, 1, 1 | 2 | 1 | 781818300, 861677900, 823489000 | **823,489,000** | 13.725 | 3.431204166666667 |
| C_line_padded | 8 | 0, 64, 128, 192 | 0, 1, 2, 3 | 4 | 2 | 571829400, 542815600, 529185000 | **542,815,600** | 9.047 | 2.2617316666666665 |
| D_prefetch_safe | 16 | 0, 128, 256, 384 | 0, 2, 4, 6 | 4 | 4 | 578777500, 607367700, 550847900 | **578,777,500** | 9.646 | 2.4115729166666666 |
| private_register | 1 | 0, 8, 16, 24 | 0, 0, 0, 0 | 1 | 1 | 44136400, 45225000, 44455300 | **44,455,300** | 0.741 | 0.18523041666666668 |

Slowdown ratios: stride1/stride8 = **2.060×**, stride4/stride8 = **1.517×**, stride16/stride8 = **1.066×**, stride1/private = **25.158×**, stride8/private = **12.210×**.

---

## Q1: MESI Protocol State Transitions

With stride = 1, the four counters sit at byte offsets 0, 8, 16, 24. My manifest gives them all relative cache-line index 0 (`distinct_cache_lines = 1`), so `buffer[0]` (Core 0) and `buffer[1]` (Core 1) are in **one 64-byte line**.

1. **Core 0 stores to `buffer[0]`.** Its load misses, and no other core has the line, so it is filled in **Exclusive (E)**. The store then upgrades it silently, **E → Modified (M)**.
2. **Core 1 stores to `buffer[1]` right after.** Core 1 doesn't own the line (its copy is **Invalid (I)**), so the store goes into Core 1's **store buffer** and Core 1 issues a **Read-For-Ownership (RFO)** for the entire line.
3. **The RFO snoops Core 0.** Core 0 holds the line in **M**, so it supplies or writes back the dirty data and moves **M → I**. Core 1 receives exclusive ownership, **I → M**, and only then can its store drain from the store buffer into L1.
4. **Next iteration.** Core 0 stores to `buffer[0]` again, finds its copy **Invalid**, and sends its own RFO. Core 1 goes **M → I** and Core 0 goes **I → M**. With 4 threads the line keeps bouncing between the cores ("ping-pong"), although no two threads ever touch the same address.

**Why the RFO stalls Core 1's store buffer:** a store can only commit to L1 once the core owns the line in E or M. During the RFO round trip (snoop, invalidation of the peer copy, cross-core data transfer), the store waits in the store buffer. My loop does a load and then a store to the same line every iteration, so the next iteration's load also waits for the line to come back. The buffer fills and the core stalls on coherence instead of doing arithmetic. That is why stride 1 costs **18.640 ns per update per thread**, compared with **9.047 ns** for stride 8.

---

## Q2: Quantitative Slowdown & Coherency Overhead

Measured median wall-clock durations (4 threads concurrently, 60,000,000 updates each):

| Regime | Median wall (ns) | ns / update / thread | ns / update (aggregate, wall / 240,000,000) |
|---|---|---|---|
| stride = 1 | 1,118,423,200 | 18.640 | 4.660096666666667 |
| stride = 4 | 823,489,000 | 13.725 | 3.431204166666667 |
| stride = 8 | 542,815,600 | 9.047 | 2.2617316666666665 |
| stride = 16 | 578,777,500 | 9.646 | 2.4115729166666666 |
| private register | 44,455,300 | 0.741 | 0.18523041666666668 |

**Slowdown of stride = 1 relative to stride = 8: 2.060×** (`stride1_over_stride8 = 2.0604109388160547`).
Other ratios: stride4/stride8 = 1.517×, stride1/private = 25.158×, stride8/private = 12.210×.

Interpretation:
- **Stride 8** (4 distinct lines) shows the cost of keeping the counter in memory when each line stays in its own core's L1: one L1 load and one L1 store per update, with store-to-load forwarding (9.047 ns/update/thread).
- **The private baseline** removes the per-iteration memory traffic altogether. The loop runs in registers, and the LLVM audit shows it was even vectorized (0 loads/stores in the loop). It runs at 0.741 ns/update/thread, which is 12.210× faster than stride 8. This factor isolates the **L1 store/forwarding latency** that the in-memory loop pays.
- **Stride 1** (1 shared line) adds the cross-core **coherence latency** on top of that: 18.640 ns/update/thread, which is 2.060× stride 8 and 25.158× private.
- **Stride 4** (2 lines, 2 threads per line, relative line indices 0, 0, 1, 1) lands in between at 13.725 ns, 1.517× stride 8. Each line now ping-pongs between only 2 cores instead of 4.

The order **stride 1 > stride 4 > stride 8 ≈ stride 16 ≫ private** matches the false-sharing model. All five configurations produced bit-identical counters ([83437500, 94687500, 92812500, 89062500]) in all 3 repeats, so the difference is purely a timing effect.

Notes on my platform:
- Context switches are `null` in the manifest. Python's `resource` module doesn't exist on Windows, so `getrusage` couldn't be called. The steady medians and tight run-to-run spread (e.g. stride 8: 529,185,000 to 571,829,400 ns) don't suggest OS scheduling as the cause.
- `physical_cores` is `null` because Windows exposes no core-sibling topology through sysfs. Windows usually numbers hyper-threaded siblings next to each other, so the pinned CPUs [0, 1, 2, 3] may be two SMT pairs. Two sibling threads share one L1, which would make some of the stride-1 sharing cheaper than true cross-core RFOs. That may be why my slowdown (2.060×) is moderate rather than extreme.

---

## Q3: L2 Spatial Prefetcher & 128-Byte Pairing

- stride 8 (64 B spacing): median **542,815,600 ns** (9.047 ns/update/thread), runs 571829400 / 542815600 / 529185000
- stride 16 (128 B spacing): median **578,777,500 ns** (9.646 ns/update/thread), runs 578777500 / 607367700 / 550847900
- ratio stride16/stride8 = **1.066×**

**How 128-byte pairing works:** Intel's L2 **adjacent-cache-line (spatial) prefetcher** treats memory as 128-byte-aligned pairs of 64-byte lines. When it brings in line *2n*, it also tries to fetch its "buddy" line *2n+1* (or the reverse).

With stride 8 in my run, the four counters are in relative lines 0, 1, 2, 3, which form **2 distinct 128-byte sectors** (`distinct_128B_sectors = 2`). Threads 0 and 1 share one sector, and threads 2 and 3 share the other. With stride 16, the counters are in lines 0, 2, 4, 6, which form **4 distinct 128-byte sectors**, so every thread owns a whole pair.

**When 64-byte padding still suffers secondary invalidations:** when a core actually misses on its line, the prefetcher may also request the buddy line, which another core is holding in **Modified**. That request snoops the other core and can downgrade or invalidate its line, even though no data is shared. This happens when lines are being (re)fetched: cold start, eviction, or after a thread is descheduled or migrated. It also requires the prefetcher to be enabled and the two threads to be on different physical cores, since SMT siblings share the L1 anyway.

**What my run shows:** stride 16 was *not* faster. It was 1.066× the stride-8 time, and the run ranges overlap (stride 8: 529,185,000 to 571,829,400 ns; stride 16: 550,847,900 to 607,367,700 ns). So the difference is within run-to-run noise. In this tight loop, each line stays resident in Modified state in its owner's L1 after the first touch, so there are almost no misses for the adjacent-line prefetcher to act on. On this machine, 64-byte padding was enough to remove false sharing, and the extra 128-byte padding gave no measurable benefit.

---

## Q4: Compiler Register Promotion vs. Volatile Memory Stores

LLVM audit of `worker_strided_accumulate.inspect_llvm()`:

- Native function: `define noundef i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(...)`
- nopython signature: `(Array(uint64, 1, 'C', False, aligned=True), int64, int64, uint64) -> uint64`
- Python object calls in the native function: **0**; volatile operations: **0**
- Basic blocks: 9; loop blocks: `B118`, `B118.epil`
- **i64 stores: 6 total, 5 inside the loop**
- **i64 loads: 6 total, 5 inside the loop**
- Loop stores:
  - `store i64 %.192, ptr %.221, align 8`
  - `store i64 %.192.1, ptr %.221.1, align 8`
  - `store i64 %.192.2, ptr %.221.2, align 8`
  - `store i64 %.192.3, ptr %.221.3, align 8`
  - `store i64 %.192.epil, ptr %.221.epil, align 8`
- Loop loads:
  - `%.180 = load i64, ptr %sunkaddr1, align 8`, plus `%.180.1`, `%.180.2`, `%.180.3` from the same `%sunkaddr1`
  - `%.180.epil = load i64, ptr %sunkaddr3, align 8`

The `.1`, `.2`, `.3` suffixes show LLVM **unrolled the loop 4×**, with an `.epil` remainder block. Each unrolled copy still has its own real `load i64` and `store i64`, so every iteration does one memory read and one memory write.

**Contrast, the private kernel:** `worker_private_accumulate` has 2 i64 stores and 1 i64 load in total, but **0 stores and 0 loads inside its loop**. Its loop blocks are `vector.body`, `vec.epilog.vector.body` and `B90`. LLVM kept the accumulator in (vector) registers and wrote memory only after the loop.

**Why an optimizing compiler would hoist `buffer[slot]`:** in a naive `for k: buffer[slot] += inc`, the address `buffer + 8·slot` is **loop-invariant**, and no other store or call in the loop could alias it. LLVM's LICM pass then applies **scalar promotion**: one load before the loop, the running sum in a register, one store after the loop. That's exactly what happened to the private kernel, and it would remove every per-iteration store, and with them all the coherence traffic I'm trying to measure.

**How my loop's data dependency prevents it:** each iteration computes

```python
v   = buffer[slot]
idx = slot + np.int64(v >> 63)   # 0 at runtime because counters stay far below 2**63
buffer[idx] = v + bit + 1
```

1. The **store address `%.221` depends on `v`**, the value loaded in that same iteration, so it is not loop-invariant. Scalar promotion needs an invariant, must-alias pointer, so LICM can't sink the store out of the loop.
2. The store to `buffer[idx]` **may alias** `buffer[slot]` as far as LLVM knows, so the next load from `%sunkaddr1` can't be replaced with the value kept in a register. It is re-issued from memory every iteration (visible as four separate `load i64, ptr %sunkaddr1` in the unrolled body).

So every iteration issues a real L1 load and store without `volatile`. That's why the strided kernel takes 542,815,600 ns even with no sharing (stride 8), while the promoted private kernel takes 44,455,300 ns.

---

## Q5: CPython GIL Masking vs. nogil=True Execution

My contrast run (stride = 1, 1,000,000 updates per thread, `sys.getswitchinterval() = 0.005` s, `gil_enabled = true` on Python 3.14.0):

| Mode | 1 thread (ns) | 4 threads (ns) | 4 threads / 1 thread |
|---|---|---|---|
| Pure Python threads | 295,584,700 | 1,170,067,100 | **3.958×** |
| Numba `@njit(nogil=True)` | 13,385,200 | 18,598,600 | **1.389×** |

Both pure-Python runs produced correct counters (`values_correct = true`).

**Why pure-Python threads show no false sharing and get slower with more threads:** only the thread holding the **Global Interpreter Lock** executes bytecode, and the interpreter forces a hand-off roughly every 0.005 s. The four Python threads therefore **never store at the same time**, so the cache line is not contended across cores in any meaningful way. Each update also costs dozens of nanoseconds of interpreter work: bytecode dispatch, integer boxing, reference counting, list indexing. With 4 threads the total work is 4× larger and still **serialized**, so wall time grows almost linearly: 295,584,700 ns to 1,170,067,100 ns (3.958×). On top of that comes the overhead of GIL contention: lock hand-offs, waking waiting threads, and OS context switches between them. (Context-switch counts are `null` here because `resource.getrusage` doesn't exist on Windows.)

**Why `nogil=True` exposes hardware contention instead:** the Numba kernel is native machine code and releases the GIL, so the four threads run **truly in parallel** on CPUs 0–3. Four threads did four times the work in only 1.389× the single-thread time (13,385,200 ns to 18,598,600 ns), so the threads overlapped instead of taking turns. The extra cost that remains is **hardware cache-line contention**: concurrent stores to the same 64-byte line trigger RFO ping-pong. The full-size benchmark measures this directly. Stride 1 (1,118,423,200 ns) is 2.060× slower than stride 8 (542,815,600 ns), where each counter has its own line.

**In short:** with the GIL, the bottleneck is a **software lock** that serializes the threads and hides the hardware effect. With `nogil=True`, the threads really share the hardware, and the bottleneck becomes the **cache-coherence protocol**, which padding to stride ≥ 8 removes.