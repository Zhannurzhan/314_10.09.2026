"""
hybrid_mpi_cuda_cache.py
------------------------
Hybrid MPI-CUDA distributed 2D stencil + MPI-3 shared-memory cache-line benchmark.

Launch (from the notebook, Cell 4):
    mpirun --allow-run-as-root --oversubscribe -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>

Map of the lab tasks:
    Task 1 -> init_hybrid_context
    Task 2 -> initialize_fields, pack_halos_kernel, unpack_halos_kernel
    Task 3 -> tiled_stencil_kernel, boundary_strip_kernel, step_serialized, step_overlapped
    Task 4 -> host_shm_accumulate, benchmark_shm_strides
    Task 5 -> main (verification, audits, manifest)
"""
import os
import re
import sys
import json
import math
import time
import hashlib
import platform
import subprocess

import numpy as np
from mpi4py import MPI
import mpi4py
import numba
from numba import cuda, njit, float64, uint64, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir as llir

# --------------------------------------------------------------------------
# Constants (from the starter notebook)
# --------------------------------------------------------------------------
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2          # 18x18 shared tile (16x16 cells + 1-cell halo)
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15                 # CFL safety factor: dt = LAMBDA_DAMP * dx^2 / kappa_max

# Derived constants
PY, PX = 2, 2                      # 2 x 2 Cartesian process grid (rows, cols)
LOCAL_N = GLOBAL_NX // PX          # 256 interior cells per rank per direction
DX = 1.0 / (GLOBAL_NX + 1)         # dx = 1/513 (walls at x=0 and x=1)
SHM_SLOTS = 64                     # 64 uint64 slots in the shared window
SHM_STRIDES = (1, 8, 16)           # 8 B, 64 B, 128 B spacing between ranks
SHM_REPEATS = 3                    # repeats per stride, median is reported
ENERGY_CHECKPOINTS = (0, 100, 200, 300)
PARITY_TOL = 1e-12

# Message tags = direction the data travels
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 10, 11, 12, 13

# Launch geometry
HALO_THREADS = 128
HALO_BLOCKS = (LOCAL_N + HALO_THREADS - 1) // HALO_THREADS
TILE_BLOCK = (TILE_DIM, TILE_DIM)
BOUNDARY_CELLS = 4 * LOCAL_N - 4   # top + bottom rows + left/right columns without corners
BOUNDARY_BLOCKS = (BOUNDARY_CELLS + HALO_THREADS - 1) // HALO_THREADS


def _grid_for(i0, i1, j0, j1):
    """2D grid (x -> columns j, y -> rows i) covering the inclusive region."""
    nj = j1 - j0 + 1
    ni = i1 - i0 + 1
    return ((nj + TILE_DIM - 1) // TILE_DIM, (ni + TILE_DIM - 1) // TILE_DIM)


FULL_GRID = _grid_for(1, LOCAL_N, 1, LOCAL_N)              # cells 1..256
INTERIOR_GRID = _grid_for(2, LOCAL_N - 1, 2, LOCAL_N - 1)  # deep interior 2..255


# --------------------------------------------------------------------------
# Hardware identity helpers
# --------------------------------------------------------------------------
def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


# ==========================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==========================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Non-periodic 2x2 Cartesian communicator (no reordering -> cart rank == world rank)
    cart_comm = comm.Create_cart(dims=[PY, PX], periods=[False, False], reorder=False)
    coords = tuple(cart_comm.Get_coords(cart_comm.Get_rank()))   # (row, col)
    up, down = cart_comm.Shift(0, 1)       # row direction
    left, right = cart_comm.Shift(1, 1)    # column direction

    # All 4 ranks share the single T4
    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    gpu_name = comm.bcast(get_gpu_name() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}::{get_cpu_model()}"
    line_bytes = detect_cpu_line_size()

    # Deterministic 64-bit instance seed (computed on rank 0, broadcast to all)
    if rank == 0:
        digest = hashlib.sha256(
            f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}::{line_bytes}".encode("utf-8")
        ).hexdigest()
        instance_seed = int(digest[:16], 16)
    else:
        digest, instance_seed = None, None
    digest = comm.bcast(digest, root=0)
    instance_seed = comm.bcast(instance_seed, root=0)

    # Intra-node communicator + MPI-3 shared window (rank 0 owns the memory)
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    if node_comm.Get_size() != 4:
        raise RuntimeError("All 4 ranks must be on the same node for the shared window.")

    pad = 128                                  # spare bytes so we can align to 64 B
    total_bytes = SHM_SLOTS * 8 + pad
    win = MPI.Win.Allocate_shared(total_bytes if node_rank == 0 else 0, 8, comm=node_comm)
    buf, _itemsize = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint8, shape=(total_bytes,))
    offset = (-raw.ctypes.data) % 64           # first 64-byte boundary
    shm_view = raw[offset: offset + SHM_SLOTS * 8].view(np.uint64)
    if shm_view.ctypes.data % 64 != 0:
        raise RuntimeError("Shared window view is not 64-byte aligned.")
    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm,
        "rank": rank,
        "size": size,
        "cart_comm": cart_comm,
        "coords": coords,
        "neighbors": {"up": up, "down": down, "left": left, "right": right},
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "cpu_sig": cpu_sig,
        "cpu_line_bytes": line_bytes,
        "digest": digest,
        "instance_seed": instance_seed,
        "node_comm": node_comm,
        "node_rank": node_rank,
        "win": win,
        "shm_raw": raw,            # keep a reference so the view stays valid
        "shm_view": shm_view,
        "shm_align_offset": offset,
    }


# ==========================================================================
# TASK 2: Rank-local subdomain + halo pack/unpack kernels
# ==========================================================================
def initialize_fields(ctx: dict) -> dict:
    """Build (258, 258) float64 arrays for u, kappa(x,y), S(x,y) from GLOBAL
    coordinates of this rank only. No global grid is ever allocated."""
    row, col = ctx["coords"]
    n = LOCAL_N
    gi = row * n + np.arange(n + 2)      # global row index 0..513 (0 and 513 are walls)
    gj = col * n + np.arange(n + 2)      # global col index
    y = (gi * DX)[:, None]
    x = (gj * DX)[None, :]

    # Seed-dependent (but rank-independent) source position
    rng = np.random.default_rng(ctx["instance_seed"])
    xs, ys = 0.3 + 0.4 * rng.random(2)

    kappa = 1.0 + 0.5 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)   # in [0.5, 1.5]
    src = 50.0 * np.exp(-((x - xs) ** 2 + (y - ys) ** 2) / (2.0 * 0.05 ** 2))
    u = np.sin(np.pi * x) * np.sin(np.pi * y)

    # Ghost frame starts at zero; halo exchange fills it every step.
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    return {
        "u0": np.ascontiguousarray(u, dtype=np.float64),
        "kappa": np.ascontiguousarray(np.broadcast_to(kappa, (n + 2, n + 2)), dtype=np.float64),
        "src": np.ascontiguousarray(np.broadcast_to(src, (n + 2, n + 2)), dtype=np.float64),
        "source_center": (float(xs), float(ys)),
    }


@cuda.jit
def pack_halos_kernel(u, sendbuf, n):
    """Copy the 4 edge strips of the interior into one contiguous 1D buffer:
       [0:n]   top row    u[1, 1..n]   (contiguous in memory)
       [n:2n]  bottom row u[n, 1..n]   (contiguous)
       [2n:3n] left col   u[1..n, 1]   (strided: 258 * 8 = 2064 B apart)
       [3n:4n] right col  u[1..n, n]   (strided)"""
    k = cuda.grid(1)
    if k < n:
        sendbuf[k] = u[1, k + 1]
        sendbuf[n + k] = u[n, k + 1]
        sendbuf[2 * n + k] = u[k + 1, 1]
        sendbuf[3 * n + k] = u[k + 1, n]


@cuda.jit
def unpack_halos_kernel(u, recvbuf, n):
    """Scatter the received 1D buffer into ghost rows/cols 0 and n+1 (= 257):
       [0:n]   from UP    -> row 0
       [n:2n]  from DOWN  -> row n+1
       [2n:3n] from LEFT  -> col 0
       [3n:4n] from RIGHT -> col n+1"""
    k = cuda.grid(1)
    if k < n:
        u[0, k + 1] = recvbuf[k]
        u[n + 1, k + 1] = recvbuf[n + k]
        u[k + 1, 0] = recvbuf[2 * n + k]
        u[k + 1, n + 1] = recvbuf[3 * n + k]


# ==========================================================================
# TASK 3: Shared-memory tiled stencil + dual-stream pipeline
# ==========================================================================
@cuda.jit(device=True)
def _flux_update(c, nn, ss, ww, ee, kc, kn, ks, kw, ke, s, coef, dt):
    """Conservative 5-point update for u_t = div(kappa grad u) + S.
    Face diffusivity = arithmetic mean of the two neighbouring cells."""
    k_n = 0.5 * (kc + kn)
    k_s = 0.5 * (kc + ks)
    k_w = 0.5 * (kc + kw)
    k_e = 0.5 * (kc + ke)
    flux = k_n * (nn - c) + k_s * (ss - c) + k_w * (ww - c) + k_e * (ee - c)
    return c + coef * flux + dt * s


@cuda.jit
def tiled_stencil_kernel(u, un, kappa, src, coef, dt, i0, i1, j0, j1):
    """16x16 thread block updates a 16x16 patch of the region [i0..i1] x [j0..j1].
    The block first loads an 18x18 tile (patch + 1-cell halo) into shared memory."""
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bi0 = i0 + cuda.blockIdx.y * TILE_DIM      # first row of this block's patch
    bj0 = j0 + cuda.blockIdx.x * TILE_DIM      # first col of this block's patch

    # Cooperative load: 256 threads fill 324 tile entries
    k = ty * TILE_DIM + tx
    while k < SHARED_DIM * SHARED_DIM:
        si = k // SHARED_DIM
        sj = k - si * SHARED_DIM
        gi = bi0 - 1 + si
        gj = bj0 - 1 + sj
        val = 0.0
        # Only read cells the region actually needs (keeps us off the ghost
        # cells that may be written concurrently on the other stream).
        if gi >= i0 - 1 and gi <= i1 + 1 and gj >= j0 - 1 and gj <= j1 + 1:
            val = u[gi, gj]
        tile[si, sj] = val
        k += TILE_DIM * TILE_DIM

    cuda.syncthreads()   # every write to the tile must finish before anyone reads it

    i = bi0 + ty
    j = bj0 + tx
    if i <= i1 and j <= j1:
        si = ty + 1
        sj = tx + 1
        un[i, j] = _flux_update(
            tile[si, sj], tile[si - 1, sj], tile[si + 1, sj], tile[si, sj - 1], tile[si, sj + 1],
            kappa[i, j], kappa[i - 1, j], kappa[i + 1, j], kappa[i, j - 1], kappa[i, j + 1],
            src[i, j], coef, dt,
        )


@cuda.jit
def boundary_strip_kernel(u, un, kappa, src, coef, dt, n):
    """Update the 4 outer border strips (rows 1 and n, cols 1 and n) after halos arrive.
    Same arithmetic as the tiled kernel, so results are bit-identical."""
    k = cuda.grid(1)
    if k >= 4 * n - 4:
        return
    if k < n:                       # top row, cols 1..n
        i = 1
        j = k + 1
    elif k < 2 * n:                 # bottom row, cols 1..n
        i = n
        j = k - n + 1
    elif k < 3 * n - 2:             # left col, rows 2..n-1
        i = k - 2 * n + 2
        j = 1
    else:                           # right col, rows 2..n-1
        i = k - (3 * n - 2) + 2
        j = n
    un[i, j] = _flux_update(
        u[i, j], u[i - 1, j], u[i + 1, j], u[i, j - 1], u[i, j + 1],
        kappa[i, j], kappa[i - 1, j], kappa[i + 1, j], kappa[i, j - 1], kappa[i, j + 1],
        src[i, j], coef, dt,
    )


def build_gpu_state(fields: dict) -> dict:
    n = LOCAL_N
    st = {
        "d_u": cuda.to_device(fields["u0"]),
        "d_un": cuda.to_device(fields["u0"]),
        "d_kappa": cuda.to_device(fields["kappa"]),
        "d_src": cuda.to_device(fields["src"]),
        "d_send": cuda.to_device(np.zeros(4 * n, dtype=np.float64)),
        "d_recv": cuda.to_device(np.zeros(4 * n, dtype=np.float64)),
        "h_send": cuda.pinned_array(4 * n, dtype=np.float64),
        "h_recv": cuda.pinned_array(4 * n, dtype=np.float64),
        "stream_interior": cuda.stream(),
        "stream_boundary": cuda.stream(),
    }
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0      # stays 0 on sides facing a wall (MPI.PROC_NULL) -> u = 0 there
    return st


def reset_gpu_state(st: dict, fields: dict) -> None:
    st["d_u"].copy_to_device(fields["u0"])
    st["d_un"].copy_to_device(fields["u0"])
    st["h_recv"][:] = 0.0
    cuda.synchronize()


def exchange_halos(ctx: dict, st: dict) -> None:
    """Non-blocking, tag-matched halo exchange over pinned host buffers."""
    n = LOCAL_N
    cart = ctx["cart_comm"]
    nb = ctx["neighbors"]
    hs, hr = st["h_send"], st["h_recv"]
    reqs = [
        # receives: tag = direction the data is travelling
        cart.Irecv([hr[0:n], MPI.DOUBLE], source=nb["up"], tag=TAG_DOWN),
        cart.Irecv([hr[n:2 * n], MPI.DOUBLE], source=nb["down"], tag=TAG_UP),
        cart.Irecv([hr[2 * n:3 * n], MPI.DOUBLE], source=nb["left"], tag=TAG_RIGHT),
        cart.Irecv([hr[3 * n:4 * n], MPI.DOUBLE], source=nb["right"], tag=TAG_LEFT),
        # sends
        cart.Isend([hs[0:n], MPI.DOUBLE], dest=nb["up"], tag=TAG_UP),
        cart.Isend([hs[n:2 * n], MPI.DOUBLE], dest=nb["down"], tag=TAG_DOWN),
        cart.Isend([hs[2 * n:3 * n], MPI.DOUBLE], dest=nb["left"], tag=TAG_LEFT),
        cart.Isend([hs[3 * n:4 * n], MPI.DOUBLE], dest=nb["right"], tag=TAG_RIGHT),
    ]
    MPI.Request.Waitall(reqs)


def step_serialized(ctx: dict, st: dict, coef: float, dt: float) -> None:
    """Everything one after another on a single stream: exchange first, then compute."""
    n = LOCAL_N
    s = st["stream_boundary"]
    d_u, d_un = st["d_u"], st["d_un"]

    pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s](d_u, st["d_send"], n)
    st["d_send"].copy_to_host(st["h_send"], stream=s)
    s.synchronize()

    exchange_halos(ctx, st)

    st["d_recv"].copy_to_device(st["h_recv"], stream=s)
    unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s](d_u, st["d_recv"], n)
    tiled_stencil_kernel[FULL_GRID, TILE_BLOCK, s](
        d_u, d_un, st["d_kappa"], st["d_src"], coef, dt, 1, n, 1, n
    )
    s.synchronize()
    st["d_u"], st["d_un"] = d_un, d_u


def step_overlapped(ctx: dict, st: dict, coef: float, dt: float) -> None:
    """Deep interior [2..255]^2 runs on stream_interior WHILE the halo
    pack -> D2H -> MPI -> H2D -> unpack -> border update runs on stream_boundary."""
    n = LOCAL_N
    s_i = st["stream_interior"]
    s_b = st["stream_boundary"]
    d_u, d_un = st["d_u"], st["d_un"]

    # 1) pack + start device->host copy of the edges
    pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s_b](d_u, st["d_send"], n)
    st["d_send"].copy_to_host(st["h_send"], stream=s_b)

    # 2) launch deep interior (needs no halo data) -- runs asynchronously
    tiled_stencil_kernel[INTERIOR_GRID, TILE_BLOCK, s_i](
        d_u, d_un, st["d_kappa"], st["d_src"], coef, dt, 2, n - 1, 2, n - 1
    )

    # 3) wait only for the edge copy, then talk MPI while the GPU computes
    s_b.synchronize()
    exchange_halos(ctx, st)

    # 4) halos back to the GPU, unpack, then update the 4 border strips
    st["d_recv"].copy_to_device(st["h_recv"], stream=s_b)
    unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s_b](d_u, st["d_recv"], n)
    boundary_strip_kernel[BOUNDARY_BLOCKS, HALO_THREADS, s_b](
        d_u, d_un, st["d_kappa"], st["d_src"], coef, dt, n
    )

    s_b.synchronize()
    s_i.synchronize()
    st["d_u"], st["d_un"] = d_un, d_u


def global_energy(ctx: dict, st: dict) -> float:
    """E = sum(u^2) * dx^2 over all ranks (cart_comm.allreduce)."""
    u = st["d_u"].copy_to_host()
    local = float(np.sum(u[1:-1, 1:-1] ** 2)) * DX * DX
    return ctx["cart_comm"].allreduce(local, op=MPI.SUM)


def run_mode(ctx, st, fields, mode, coef, dt):
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    reset_gpu_state(st, fields)
    ctx["cart_comm"].Barrier()

    energies = {0: global_energy(ctx, st)}
    t_gpu = 0.0
    for step in range(1, STENCIL_STEPS + 1):
        t0 = time.perf_counter()
        step_fn(ctx, st, coef, dt)
        t_gpu += time.perf_counter() - t0
        if step in ENERGY_CHECKPOINTS:
            energies[step] = global_energy(ctx, st)   # not included in the timing

    final_u = st["d_u"].copy_to_host()
    gpu_ms = ctx["cart_comm"].allreduce(t_gpu * 1000.0, op=MPI.MAX)
    return final_u, energies, gpu_ms


# ==========================================================================
# TASK 4: Multi-rank host cache-line false-sharing sweep
# ==========================================================================
@intrinsic
def _plain_rmw_increment(typingctx, arr_ty, idx_ty):
    """arr[idx] += 1 done as an explicit load + add + store on memory.
    'monotonic' (relaxed) atomic load/store = a plain MOV on x86 (no LOCK),
    but LLVM is not allowed to keep the value in a register across loop
    iterations, so every iteration really touches the shared cache line."""
    if not isinstance(arr_ty, types.Array) or not isinstance(idx_ty, types.Integer):
        return None
    sig = types.void(arr_ty, idx_ty)

    def codegen(context, builder, signature, args):
        aty, _ity = signature.args
        arr_val, idx_val = args
        ary = context.make_array(aty)(context, builder, arr_val)
        ptr = cgutils.get_item_pointer(context, builder, aty, ary, [idx_val], wraparound=False)
        old = builder.load_atomic(ptr, "monotonic", 8)
        new = builder.add(old, llir.Constant(old.type, 1))
        builder.store_atomic(new, ptr, "monotonic", 8)
        return context.get_dummy_value()

    return sig, codegen


@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters):
    for _ in range(iters):
        _plain_rmw_increment(shm_view, slot)
    return shm_view[slot]


def benchmark_shm_strides(ctx: dict) -> dict:
    node_comm = ctx["node_comm"]
    node_rank = ctx["node_rank"]
    shm = ctx["shm_view"]
    nranks = node_comm.Get_size()
    base = shm.ctypes.data

    # JIT warm-up (compile outside of the timed region)
    host_shm_accumulate(shm, node_rank * 16, 1)
    node_comm.Barrier()

    results = {}
    for stride in SHM_STRIDES:
        slot = node_rank * stride
        times = []
        ok = True
        for _ in range(SHM_REPEATS):
            if node_rank == 0:
                shm[:] = 0
            node_comm.Barrier()
            t0 = time.perf_counter()
            host_shm_accumulate(shm, slot, SHM_ITERS)
            elapsed = time.perf_counter() - t0
            node_comm.Barrier()
            ok = ok and (int(shm[slot]) == SHM_ITERS)
            times.append(elapsed)
        local_median = sorted(times)[len(times) // 2]
        worst = node_comm.allreduce(local_median, op=MPI.MAX)
        all_ok = node_comm.allreduce(ok, op=MPI.LAND)

        addrs = [base + 8 * r * stride for r in range(nranks)]
        results[stride] = {
            "stride_slots": stride,
            "stride_bytes": 8 * stride,
            "elapsed_ms": worst * 1000.0,
            "ns_per_update": worst * 1e9 / SHM_ITERS,
            "cpu_line_index_per_rank": [(a // 64) - (base // 64) for a in addrs],
            "distinct_64B_lines": len({a // 64 for a in addrs}),
            "distinct_128B_pairs": len({a // 128 for a in addrs}),
            "counts_correct": bool(all_ok),
        }
    return results


# ==========================================================================
# Audits & telemetry
# ==========================================================================
def _join_inspect(obj) -> str:
    if isinstance(obj, dict):
        return "\n".join(str(v) for v in obj.values())
    return str(obj)


def audit_ptx() -> dict:
    ptx = _join_inspect(tiled_stencil_kernel.inspect_asm())
    return {
        "st.shared.f64": len(re.findall(r"\bst\.shared(?:\.v[24])?\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared(?:\.v[24])?\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\b(?:bar|barrier)\.sync\b", ptx)),
        "ptx_lines": ptx.count("\n"),
    }


def audit_llvm() -> dict:
    ir_text = _join_inspect(host_shm_accumulate.inspect_llvm())
    asm_text = _join_inspect(host_shm_accumulate.inspect_asm())
    load_atomic = ir_text.count("load atomic")
    store_atomic = ir_text.count("store atomic")
    return {
        "load_atomic_monotonic": load_atomic,
        "store_atomic_monotonic": store_atomic,
        "monotonic_tokens": ir_text.count("monotonic"),
        "x86_lock_prefixes": len(re.findall(r"\block\b", asm_text)),
        "register_hoisting_prevented": bool(load_atomic >= 1 and store_atomic >= 1),
    }


def coalescing_telemetry(st: dict) -> dict:
    """Apply SectorIndex_GPU(k) = floor((Base + 8k) / 32) to real device addresses."""
    n = LOCAL_N
    pitch = n + 2
    base = st["d_u"].device_ctypes_pointer.value
    send_base = st["d_send"].device_ctypes_pointer.value

    col_addrs = [base + (i * pitch + 1) * 8 for i in range(1, n + 1)]   # u[1:257, 1]
    row_addrs = [base + (1 * pitch + j) * 8 for j in range(1, n + 1)]   # u[1, 1:257]
    packed_addrs = [send_base + 8 * k for k in range(n)]                # packed strip

    def sectors(a):
        return len({x // 32 for x in a})

    def lines128(a):
        return len({x // 128 for x in a})

    useful = n * 8
    col_sec, row_sec, pk_sec = sectors(col_addrs), sectors(row_addrs), sectors(packed_addrs)
    halo_bytes = 4 * n * 8
    full_field_bytes = pitch * pitch * 8
    return {
        "row_stride_bytes": pitch * 8,
        "useful_bytes_per_strip": useful,
        "strided_column_sectors_32B": col_sec,
        "strided_column_lines_128B": lines128(col_addrs),
        "strided_column_bytes_moved": col_sec * 32,
        "strided_column_efficiency": useful / (col_sec * 32),
        "contiguous_row_sectors_32B": row_sec,
        "contiguous_row_efficiency": useful / (row_sec * 32),
        "packed_strip_sectors_32B": pk_sec,
        "packed_strip_efficiency": useful / (pk_sec * 32),
        "pcie_bytes_per_step_packed_each_way": halo_bytes,
        "pcie_bytes_per_step_full_field_each_way": full_field_bytes,
        "pcie_reduction_factor": full_field_bytes / halo_bytes,
    }


# ==========================================================================
# TASK 5: Cross-tier verification & manifest export
# ==========================================================================
def main():
    github_user = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_USERNAME", "")
    if not github_user.strip() or github_user == "REPLACE_WITH_YOUR_GITHUB_USERNAME":
        raise RuntimeError("Pass your GitHub username as the first argument.")

    ctx = init_hybrid_context(github_user)
    rank = ctx["rank"]
    cart = ctx["cart_comm"]

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    log("=" * 72)
    log(f"Hybrid MPI-CUDA lab | user={github_user} | seed=0x{ctx['instance_seed']:016x}")
    log(f"GPU: {ctx['gpu_name']} | {ctx['gpu_uuid']}")
    log(f"CPU: {ctx['cpu_sig']} | coherency line = {ctx['cpu_line_bytes']} B")
    log(f"MPI: {MPI.Get_library_version().splitlines()[0]}")
    log("=" * 72)
    for r in range(ctx["size"]):
        if r == rank:
            nb = ctx["neighbors"]
            print(f"[rank {rank}] coords={ctx['coords']} node_rank={ctx['node_rank']} "
                  f"up={nb['up']} down={nb['down']} left={nb['left']} right={nb['right']} "
                  f"(-ve = PROC_NULL wall)", flush=True)
        cart.Barrier()

    # ---- Task 2: fields + stability -------------------------------------
    fields = initialize_fields(ctx)
    kappa_max = cart.allreduce(float(fields["kappa"].max()), op=MPI.MAX)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = LAMBDA_DAMP * DX * DX / kappa_max
    coef = dt / (DX * DX)
    log(f"dx={DX:.6e}  kappa_max={kappa_max:.6f}  dt={dt:.6e}  dt_max={dt_max:.6e}  "
        f"dt/dt_max={dt / dt_max:.3f}  CFL_ok={dt <= dt_max}")

    st = build_gpu_state(fields)

    # ---- warm-up: compile every kernel once (not timed) ------------------
    step_serialized(ctx, st, coef, dt)
    step_overlapped(ctx, st, coef, dt)
    cuda.synchronize()

    # ---- Task 3/5: the two execution modes -------------------------------
    u_ser, e_ser, gpu_serialized_ms = run_mode(ctx, st, fields, "serialized", coef, dt)
    u_ovl, e_ovl, gpu_overlapped_ms = run_mode(ctx, st, fields, "overlapped", coef, dt)

    local_diff = float(np.max(np.abs(u_ser[1:-1, 1:-1] - u_ovl[1:-1, 1:-1])))
    max_diff = cart.allreduce(local_diff, op=MPI.MAX)
    parity_ok = max_diff < PARITY_TOL

    energy_rel_diff = {
        k: abs(e_ser[k] - e_ovl[k]) / max(abs(e_ser[k]), 1e-300) for k in ENERGY_CHECKPOINTS
    }
    energy_ok = all(math.isfinite(v) for v in e_ser.values()) and \
        max(energy_rel_diff.values()) < PARITY_TOL

    log("-" * 72)
    log(f"gpu_serialized_ms = {gpu_serialized_ms:.3f}")
    log(f"gpu_overlapped_ms = {gpu_overlapped_ms:.3f}")
    log(f"overlap speedup   = {gpu_serialized_ms / gpu_overlapped_ms:.3f}x")
    log(f"max_diff (serialized vs overlapped, all ranks) = {max_diff:.3e}  PASS={parity_ok}")
    for k in ENERGY_CHECKPOINTS:
        log(f"  allreduce energy step {k:3d}: serialized={e_ser[k]:.15e}  "
            f"overlapped={e_ovl[k]:.15e}  rel_diff={energy_rel_diff[k]:.2e}")

    # ---- GPU coalescing telemetry (Q1) ------------------------------------
    coal = coalescing_telemetry(st)
    log("-" * 72)
    log(f"Strided column u[1:257,1]: {coal['strided_column_sectors_32B']} sectors x 32 B = "
        f"{coal['strided_column_bytes_moved']} B for {coal['useful_bytes_per_strip']} useful B "
        f"(efficiency {coal['strided_column_efficiency']:.1%})")
    log(f"Packed strip: {coal['packed_strip_sectors_32B']} sectors "
        f"(efficiency {coal['packed_strip_efficiency']:.1%})")
    log(f"PCIe per step each way: packed={coal['pcie_bytes_per_step_packed_each_way']} B vs "
        f"full field={coal['pcie_bytes_per_step_full_field_each_way']} B "
        f"({coal['pcie_reduction_factor']:.1f}x less)")

    # ---- PTX audit (Q2) ---------------------------------------------------
    ptx_counts = audit_ptx() if rank == 0 else None
    log("-" * 72)
    log(f"PTX audit tiled_stencil_kernel: {ptx_counts}")

    # ---- Task 4: shared-window false-sharing sweep (Q4, Q5) ---------------
    shm_results = benchmark_shm_strides(ctx)
    llvm_counts = audit_llvm() if rank == 0 else None
    t1 = shm_results[1]["elapsed_ms"]
    t8 = shm_results[8]["elapsed_ms"]
    t16 = shm_results[16]["elapsed_ms"]
    log("-" * 72)
    for s in SHM_STRIDES:
        r = shm_results[s]
        log(f"stride={s:2d} ({r['stride_bytes']:3d} B): {r['elapsed_ms']:9.2f} ms  "
            f"{r['ns_per_update']:6.2f} ns/update  lines/rank={r['cpu_line_index_per_rank']}  "
            f"distinct 64B lines={r['distinct_64B_lines']}  128B pairs={r['distinct_128B_pairs']}  "
            f"counts_ok={r['counts_correct']}")
    log(f"false-sharing slowdown (stride1/stride8)  = {t1 / t8:.2f}x")
    log(f"prefetch-pair ratio    (stride8/stride16) = {t8 / t16:.3f}x")
    log(f"LLVM IR audit host_shm_accumulate: {llvm_counts}")

    # ---- manifest ---------------------------------------------------------
    if rank == 0:
        topology = []
        for r in range(ctx["size"]):
            cy, cx = cart.Get_coords(r)
            topology.append({"rank": r, "coords": [cy, cx]})
        manifest = {
            "github_username": github_user,
            "instance_seed_hex": f"0x{ctx['instance_seed']:016x}",
            "identity_digest_sha256": ctx["digest"],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": {
                "gpu_name": ctx["gpu_name"],
                "gpu_uuid": ctx["gpu_uuid"],
                "cpu_signature": ctx["cpu_sig"],
                "cpu_coherency_line_bytes": ctx["cpu_line_bytes"],
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "numba": numba.__version__,
                "mpi4py": mpi4py.__version__,
                "mpi_library": MPI.Get_library_version().splitlines()[0],
            },
            "mpi": {"world_size": ctx["size"], "cart_dims": [PY, PX],
                    "periodic": [False, False], "topology": topology},
            "grid": {"global": [GLOBAL_NY, GLOBAL_NX], "local_interior": [LOCAL_N, LOCAL_N],
                     "local_padded": [LOCAL_N + 2, LOCAL_N + 2], "steps": STENCIL_STEPS},
            "stability": {"dx": DX, "kappa_max": kappa_max, "dt": dt, "dt_max": dt_max,
                          "dt_over_dt_max": dt / dt_max, "lambda_damp": LAMBDA_DAMP,
                          "cfl_satisfied": dt <= dt_max},
            "source_center": fields["source_center"],
            "timing": {"gpu_serialized_ms": gpu_serialized_ms,
                       "gpu_overlapped_ms": gpu_overlapped_ms,
                       "overlap_speedup": gpu_serialized_ms / gpu_overlapped_ms},
            "verification": {"max_diff": max_diff, "tolerance": PARITY_TOL,
                             "parity_pass": parity_ok, "energy_pass": energy_ok},
            "energy_allreduce": {
                "serialized": {str(k): e_ser[k] for k in ENERGY_CHECKPOINTS},
                "overlapped": {str(k): e_ovl[k] for k in ENERGY_CHECKPOINTS},
                "relative_diff": {str(k): energy_rel_diff[k] for k in ENERGY_CHECKPOINTS},
            },
            "gpu_coalescing": coal,
            "ptx_audit": ptx_counts,
            "llvm_audit": llvm_counts,
            "shm_benchmark": {
                "iters_per_rank": SHM_ITERS,
                "window_align_offset_bytes": ctx["shm_align_offset"],
                "strides": {str(s): shm_results[s] for s in SHM_STRIDES},
                "false_sharing_slowdown_s1_over_s8": t1 / t8,
                "prefetch_ratio_s8_over_s16": t8 / t16,
            },
        }
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log("-" * 72)
        log("Wrote hybrid_lab_manifest.json")

    cart.Barrier()
    ctx["win"].Free()

    if not (parity_ok and energy_ok):
        raise RuntimeError(f"Verification failed: max_diff={max_diff:.3e}, energy_ok={energy_ok}")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # make sure one failing rank does not hang the others
        print(f"[rank {MPI.COMM_WORLD.Get_rank()}] ERROR: {exc!r}", file=sys.stderr, flush=True)
        MPI.COMM_WORLD.Abort(1)
