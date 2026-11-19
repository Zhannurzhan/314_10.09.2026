import os
import sys
import json
import math
import hashlib
import subprocess
import re
from types import SimpleNamespace
from datetime import datetime, timezone
import numpy as np
from mpi4py import MPI
from numba import cuda, float64
from numba import config as numba_config

# The 1-block pack/unpack launches are intentionally tiny; silence occupancy warnings on stderr.
numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = False

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

# ---- Additional solver constants -------------------------------------------------
PROC_DIMS = [2, 2]                              # 2 x 2 Cartesian process grid
NY_LOCAL = GLOBAL_NY // PROC_DIMS[0]            # 256 owned rows per rank
NX_LOCAL = GLOBAL_NX // PROC_DIMS[1]            # 256 owned cols per rank
DX = 1.0 / (GLOBAL_NX + 1)                      # 1/513
DY = 1.0 / (GLOBAL_NY + 1)                      # 1/513
INV_DX2 = 1.0 / (DX * DX)
INV_DY2 = 1.0 / (DY * DY)
KAPPA_MIN, KAPPA_MAX = 0.08, 0.24
CFL_SAFETY = 0.5                                # dt = 0.5 * dt_max
DIAG_EVERY = 100
WARMUP_STEPS = 3
PACK_THREADS = 256
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 101, 102, 103, 104   # tag = direction of travel
SIDES = ("top", "bottom", "left", "right")


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")
    assert size == 4

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2 x 2 Cartesian communicator
    cart = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart.Get_rank()
    row_rank, col_rank = cart.Get_coords(cart_rank)

    # Shift returns (source, dest) = (coord - 1, coord + 1); PROC_NULL off the edge
    nbr_up, nbr_down = cart.Shift(0, 1)      # axis 0 = rows
    nbr_left, nbr_right = cart.Shift(1, 1)   # axis 1 = columns

    dev = cuda.get_current_device()
    gpu_name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)

    return SimpleNamespace(
        comm=comm, cart=cart, rank=cart_rank, world_rank=rank, size=size,
        row_rank=int(row_rank), col_rank=int(col_rank),
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, gpu_name=gpu_name,
        compute_capability=tuple(int(c) for c in dev.compute_capability),
        instance_seed=instance_seed, github_user=github_user.strip(),
        ny=NY_LOCAL, nx=NX_LOCAL,
    )


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (no global array)
# ==============================================================================
def initialize_subdomains(ctx):
    ny, nx = ctx.ny, ctx.nx
    # Global index of local index 0..257: g = rank_coord * n + local.
    # g = 0 and g = 513 are the physical boundaries (y=0, y=1 / x=0, x=1).
    gi = ctx.row_rank * ny + np.arange(ny + 2, dtype=np.float64)
    gj = ctx.col_rank * nx + np.arange(nx + 2, dtype=np.float64)
    y = gi * DY                     # rows    -> y
    x = gj * DX                     # columns -> x
    Y, X = np.meshgrid(y, x, indexing="ij")   # local (258, 258) only

    # Same seed on every rank -> identical random stream -> globally consistent fields
    rng = np.random.default_rng(ctx.instance_seed)

    # Diffusivity kappa(x, y) in [0.08, 0.24]
    phx, phy = rng.uniform(0.0, 1.0, size=2)
    kx, ky = rng.integers(1, 4, size=2)
    k_mid = 0.5 * (KAPPA_MIN + KAPPA_MAX)
    k_amp = 0.5 * (KAPPA_MAX - KAPPA_MIN)
    kappa = k_mid + k_amp * np.sin(2.0 * np.pi * (kx * X + phx)) * np.cos(2.0 * np.pi * (ky * Y + phy))

    # Seeded Gaussian thermal pulses
    n_pulses = 3
    centers = rng.uniform(0.2, 0.8, size=(n_pulses, 2))
    amps = rng.uniform(0.8, 1.6, size=n_pulses)
    widths = rng.uniform(0.04, 0.08, size=n_pulses)
    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    for (cx, cy), a, w in zip(centers, amps, widths):
        u += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * w * w))

    # Localized (weak) source term
    sx, sy = rng.uniform(0.3, 0.7, size=2)
    s_amp = rng.uniform(0.01, 0.03)
    s_w = 0.03
    src = s_amp * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2.0 * s_w * s_w))

    # Homogeneous Dirichlet u = 0 on physical (PROC_NULL) edges
    if ctx.nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx.nbr_down == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx.nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx.nbr_right == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return SimpleNamespace(
        h_u0=np.ascontiguousarray(u),
        h_kap=np.ascontiguousarray(kappa),
        h_src=np.ascontiguousarray(src),
        local_kappa_max=float(kappa.max()),
        local_kappa_min=float(kappa.min()),
        pulses=[{"cx": float(c[0]), "cy": float(c[1]), "amp": float(a), "sigma": float(w)}
                for c, a, w in zip(centers, amps, widths)],
        source={"sx": float(sx), "sy": float(sy), "amp": float(s_amp), "sigma": s_w},
    )


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, s_top, s_bot, s_left, s_right):
    k = cuda.grid(1)
    ny = u.shape[0] - 2
    nx = u.shape[1] - 2
    if k < nx:
        s_top[k] = u[1, k + 1]          # owned row 1 (contiguous)
        s_bot[k] = u[ny, k + 1]         # owned row ny (contiguous)
    if k < ny:
        s_left[k] = u[k + 1, 1]         # owned col 1  (stride = nx + 2)
        s_right[k] = u[k + 1, nx]       # owned col nx (stride = nx + 2)


@cuda.jit
def unpack_halos_kernel(u, r_top, r_bot, r_left, r_right,
                        has_up, has_down, has_left, has_right):
    k = cuda.grid(1)
    ny = u.shape[0] - 2
    nx = u.shape[1] - 2
    if k < nx:
        u[0, k + 1] = r_top[k] if has_up != 0 else 0.0
        u[ny + 1, k + 1] = r_bot[k] if has_down != 0 else 0.0
    if k < ny:
        u[k + 1, 0] = r_left[k] if has_left != 0 else 0.0
        u[k + 1, nx + 1] = r_right[k] if has_right != 0 else 0.0


def allocate_halo_buffers(ctx):
    n = {"top": ctx.nx, "bottom": ctx.nx, "left": ctx.ny, "right": ctx.ny}
    hb = SimpleNamespace(d_send={}, d_recv={}, h_send={}, h_recv={})
    for s in SIDES:
        hb.d_send[s] = cuda.device_array(n[s], dtype=np.float64)
        hb.d_recv[s] = cuda.to_device(np.zeros(n[s], dtype=np.float64))
        hb.h_send[s] = cuda.pinned_array(n[s], dtype=np.float64)   # page-locked
        hb.h_recv[s] = cuda.pinned_array(n[s], dtype=np.float64)   # page-locked
        hb.h_send[s][:] = 0.0
        hb.h_recv[s][:] = 0.0
    hb.blocks = (max(ctx.nx, ctx.ny) + PACK_THREADS - 1) // PACK_THREADS
    hb.flags = (
        int(ctx.nbr_up != MPI.PROC_NULL), int(ctx.nbr_down != MPI.PROC_NULL),
        int(ctx.nbr_left != MPI.PROC_NULL), int(ctx.nbr_right != MPI.PROC_NULL),
    )
    return hb


def exchange_halos_async(ctx, hb, d_u, stream):
    """Stage A (GPU pack + async D2H into pinned buffers) and posting of Stage B MPI calls.
    Returns the list of outstanding MPI requests."""
    pack_halos_kernel[hb.blocks, PACK_THREADS, stream](
        d_u, hb.d_send["top"], hb.d_send["bottom"], hb.d_send["left"], hb.d_send["right"])
    for s in SIDES:
        hb.d_send[s].copy_to_host(hb.h_send[s], stream=stream)   # async DMA (pinned)
    stream.synchronize()                                          # host send buffers ready

    cart = ctx.cart
    reqs = [
        # receive: tag = direction the neighbour's message travels
        cart.Irecv([hb.h_recv["top"], MPI.DOUBLE], source=ctx.nbr_up, tag=TAG_DOWN),
        cart.Irecv([hb.h_recv["bottom"], MPI.DOUBLE], source=ctx.nbr_down, tag=TAG_UP),
        cart.Irecv([hb.h_recv["left"], MPI.DOUBLE], source=ctx.nbr_left, tag=TAG_RIGHT),
        cart.Irecv([hb.h_recv["right"], MPI.DOUBLE], source=ctx.nbr_right, tag=TAG_LEFT),
        cart.Isend([hb.h_send["top"], MPI.DOUBLE], dest=ctx.nbr_up, tag=TAG_UP),
        cart.Isend([hb.h_send["bottom"], MPI.DOUBLE], dest=ctx.nbr_down, tag=TAG_DOWN),
        cart.Isend([hb.h_send["left"], MPI.DOUBLE], dest=ctx.nbr_left, tag=TAG_LEFT),
        cart.Isend([hb.h_send["right"], MPI.DOUBLE], dest=ctx.nbr_right, tag=TAG_RIGHT),
    ]
    return reqs


def complete_halo_exchange(hb, d_u, reqs, stream):
    """Stage C (first half): Waitall, async H2D of pinned halos, GPU unpack into ghosts.
    Returns seconds spent blocked in Waitall."""
    t0 = MPI.Wtime()
    MPI.Request.Waitall(reqs)
    waited = MPI.Wtime() - t0
    for s in SIDES:
        hb.d_recv[s].copy_to_device(hb.h_recv[s], stream=stream)  # async DMA (pinned)
    unpack_halos_kernel[hb.blocks, PACK_THREADS, stream](
        d_u, hb.d_recv["top"], hb.d_recv["bottom"], hb.d_recv["left"], hb.d_recv["right"],
        hb.flags[0], hb.flags[1], hb.flags[2], hb.flags[3])
    return waited


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kap, src, dt, inv_dx2, inv_dy2, lam,
                         r_start, r_end, c_start, c_end):
    s_u = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    s_k = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # The launch grid only covers the tiles that intersect the bounding box
    by0 = (r_start - 1) // TILE_DIM
    bx0 = (c_start - 1) // TILE_DIM
    i = (cuda.blockIdx.y + by0) * TILE_DIM + ty + 1
    j = (cuda.blockIdx.x + bx0) * TILE_DIM + tx + 1
    sy = ty + 1
    sx = tx + 1

    # Only load cells this launch actually needs (box + 1-cell rim). This keeps the
    # deep-interior launch from touching ghost cells being unpacked on the other stream.
    lo_r = r_start - 1
    hi_r = r_end + 1
    lo_c = c_start - 1
    hi_c = c_end + 1

    if i >= lo_r and i <= hi_r and j >= lo_c and j <= hi_c:
        s_u[sy, sx] = u[i, j]                       # centre cell
        s_k[sy, sx] = kap[i, j]
        if ty == 0 and i - 1 >= lo_r:               # top apron
            s_u[0, sx] = u[i - 1, j]
            s_k[0, sx] = kap[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 <= hi_r:    # bottom apron
            s_u[SHARED_DIM - 1, sx] = u[i + 1, j]
            s_k[SHARED_DIM - 1, sx] = kap[i + 1, j]
        if tx == 0 and j - 1 >= lo_c:               # left apron
            s_u[sy, 0] = u[i, j - 1]
            s_k[sy, 0] = kap[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 <= hi_c:    # right apron
            s_u[sy, SHARED_DIM - 1] = u[i, j + 1]
            s_k[sy, SHARED_DIM - 1] = kap[i, j + 1]

    cuda.syncthreads()   # every thread reaches this barrier (no early return above)

    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        uc = s_u[sy, sx]
        kc = s_k[sy, sx]
        u_n = s_u[sy - 1, sx]
        u_s = s_u[sy + 1, sx]
        u_w = s_u[sy, sx - 1]
        u_e = s_u[sy, sx + 1]
        k_n = 0.5 * (s_k[sy - 1, sx] + kc)   # kappa at (i-1/2, j)
        k_s = 0.5 * (s_k[sy + 1, sx] + kc)   # kappa at (i+1/2, j)
        k_w = 0.5 * (s_k[sy, sx - 1] + kc)   # kappa at (i, j-1/2)
        k_e = 0.5 * (s_k[sy, sx + 1] + kc)   # kappa at (i, j+1/2)
        # Spec pairs index i with dx^2 and j with dy^2; dx == dy here so labels are interchangeable.
        div_i = (k_s * (u_s - uc) - k_n * (uc - u_n)) * inv_dx2
        div_j = (k_e * (u_e - uc) - k_w * (uc - u_w)) * inv_dy2
        u_new[i, j] = uc + dt * (div_i + div_j - lam * uc * uc * uc + src[i, j])


def launch_stencil(stream, F, d_u, d_un, dt, r0, r1, c0, c1):
    """Launch the tiled stencil only over tiles covering rows r0..r1, cols c0..c1 (inclusive)."""
    by0, by1 = (r0 - 1) // TILE_DIM, (r1 - 1) // TILE_DIM
    bx0, bx1 = (c0 - 1) // TILE_DIM, (c1 - 1) // TILE_DIM
    grid = (bx1 - bx0 + 1, by1 - by0 + 1)        # (x = columns, y = rows)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_u, d_un, F.d_kap, F.d_src, dt, INV_DX2, INV_DY2, LAMBDA_DAMP, r0, r1, c0, c1)


def step_serialized(ctx, F, hb, S, d_u, d_un, dt):
    """Reference: exchange fully, THEN compute the whole subdomain on one stream."""
    s = S.boundary
    reqs = exchange_halos_async(ctx, hb, d_u, s)
    waited = complete_halo_exchange(hb, d_u, reqs, s)
    launch_stencil(s, F, d_u, d_un, dt, 1, ctx.ny, 1, ctx.nx)
    s.synchronize()
    return waited


def step_overlapped(ctx, F, hb, S, d_u, d_un, dt):
    """Dual-stream pipeline: deep interior on stream_interior overlaps the MPI exchange."""
    sb, si = S.boundary, S.interior
    ny, nx = ctx.ny, ctx.nx
    # Stage A + MPI posting (Stage B part 1)
    reqs = exchange_halos_async(ctx, hb, d_u, sb)
    # Stage B: deep interior [2..ny-1, 2..nx-1] needs no ghost cells
    launch_stencil(si, F, d_u, d_un, dt, 2, ny - 1, 2, nx - 1)
    # Stage C: wait, H2D, unpack, then the 1-cell perimeter ring
    waited = complete_halo_exchange(hb, d_u, reqs, sb)
    launch_stencil(sb, F, d_u, d_un, dt, 1, 1, 1, nx)            # row 1
    launch_stencil(sb, F, d_u, d_un, dt, ny, ny, 1, nx)          # row ny
    launch_stencil(sb, F, d_u, d_un, dt, 2, ny - 1, 1, 1)        # col 1  (corners done above)
    launch_stencil(sb, F, d_u, d_un, dt, 2, ny - 1, nx, nx)      # col nx
    sb.synchronize()
    si.synchronize()
    return waited


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def global_diagnostics(ctx, d_u, d_prev, dt):
    u = d_u.copy_to_host()[1:-1, 1:-1]
    loc = np.zeros(3, dtype=np.float64)
    loc[0] = u.sum()
    loc[1] = np.square(u).sum()
    if d_prev is not None:
        p = d_prev.copy_to_host()[1:-1, 1:-1]
        loc[2] = np.square(u - p).sum()
    glob = np.zeros(3, dtype=np.float64)
    ctx.cart.Allreduce(loc, glob, op=MPI.SUM)          # scalars only, never field arrays
    energy = float(glob[0] * DX * DY)
    l2 = float(math.sqrt(glob[1] * DX * DY))
    resid = float(math.sqrt(glob[2] * DX * DY) / dt) if d_prev is not None else None
    return energy, l2, resid


def run_mode(mode, ctx, F, hb, S, dt, n_steps, record=True):
    step_fn = step_overlapped if mode == "overlapped" else step_serialized
    F.d_u.copy_to_device(F.h_u0)
    F.d_un.copy_to_device(F.h_u0)
    cuda.synchronize()
    d_u, d_un = F.d_u, F.d_un

    trace = []
    if record:
        e, l2, r = global_diagnostics(ctx, d_u, None, dt)
        trace.append({"step": 0, "energy": e, "l2_norm": l2, "residual_l2": r})

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    gpu_ms, wall_s, wait_s, step = 0.0, 0.0, 0.0, 0
    while step < n_steps:
        seg = min(DIAG_EVERY, n_steps - step)
        ctx.cart.Barrier()
        t0 = MPI.Wtime()
        start.record(stream=S.boundary)
        for _ in range(seg):
            wait_s += step_fn(ctx, F, hb, S, d_u, d_un, dt)
            d_u, d_un = d_un, d_u                       # swap double buffers
        end.record(stream=S.boundary)
        end.synchronize()
        wall_s += MPI.Wtime() - t0
        gpu_ms += cuda.event_elapsed_time(start, end)
        step += seg
        if record:                                      # diagnostics are outside the timed region
            e, l2, r = global_diagnostics(ctx, d_u, d_un, dt)
            trace.append({"step": step, "energy": e, "l2_norm": l2, "residual_l2": r})

    return SimpleNamespace(final=d_u.copy_to_host(), trace=trace,
                           gpu_ms=gpu_ms, wall_ms=wall_s * 1e3, mpi_wait_ms=wait_s * 1e3)


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        asm = "\n".join(asm.values())
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", asm)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", asm)),
        "st.shared(any)": len(re.findall(r"\bst\.shared\.", asm)),
        "ld.shared(any)": len(re.findall(r"\bld\.shared\.", asm)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", asm)),
        "barrier.sync": len(re.findall(r"\bbarrier\.sync\b", asm)),
    }
    passed = (counts["st.shared(any)"] > 0 and counts["ld.shared(any)"] > 0
              and (counts["bar.sync"] + counts["barrier.sync"]) > 0)
    return counts, passed, asm


def _fmt(v):
    return "n/a" if v is None else f"{v:.15e}"


def _nbr(r):
    return "NULL" if r == MPI.PROC_NULL else str(r)


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    root = ctx.rank == 0

    # ---- Task 2: local fields only -------------------------------------------
    F = initialize_subdomains(ctx)
    kbuf = np.array([F.local_kappa_max], dtype=np.float64)
    kmax = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(kbuf, kmax, op=MPI.MAX)
    kminl = np.array([F.local_kappa_min], dtype=np.float64)
    kmin = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(kminl, kmin, op=MPI.MIN)
    kappa_max = float(kmax[0])
    kappa_min = float(kmin[0])
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    F.d_u = cuda.to_device(F.h_u0)          # (258, 258)
    F.d_un = cuda.to_device(F.h_u0)         # (258, 258) double buffer
    F.d_kap = cuda.to_device(F.h_kap)
    F.d_src = cuda.to_device(F.h_src)
    hb = allocate_halo_buffers(ctx)
    S = SimpleNamespace(boundary=cuda.stream(), interior=cuda.stream())   # stream_boundary / stream_interior

    # ---- Topology table via Allreduce on small int metadata (no gather) -------
    topo_local = np.zeros((ctx.size, 7), dtype=np.int64)
    topo_local[ctx.rank] = [ctx.rank, ctx.row_rank, ctx.col_rank,
                            ctx.nbr_up, ctx.nbr_down, ctx.nbr_left, ctx.nbr_right]
    topo = np.zeros_like(topo_local)
    ctx.cart.Allreduce(topo_local, topo, op=MPI.SUM)

    if root:
        print("=" * 78)
        print("Distributed MPI + CUDA anisotropic reaction-diffusion solver")
        print("=" * 78)
        print(f"GITHUB_USERNAME     : {ctx.github_user}")
        print(f"GPU                 : {ctx.gpu_name}  (CC {ctx.compute_capability[0]}.{ctx.compute_capability[1]})")
        print(f"GPU UUID            : {ctx.gpu_uuid}")
        print(f"INSTANCE_SEED       : {ctx.instance_seed}  (0x{ctx.instance_seed:016x})")
        print(f"MPI ranks           : {ctx.size}  Cartesian dims={PROC_DIMS} periods=[False, False]")
        print(f"Global grid         : {GLOBAL_NY} x {GLOBAL_NX}   local padded: {NY_LOCAL + 2} x {NX_LOCAL + 2}")
        print(f"dx = dy             : {DX:.10e}")
        print(f"kappa range         : [{kappa_min:.6f}, {kappa_max:.6f}]")
        print(f"dt_max = dx^2/(4k)  : {dt_max:.10e}")
        print(f"dt (CFL x {CFL_SAFETY})      : {dt:.10e}")
        print("-" * 78)
        print(" rank (row,col)    up  down  left right")
        for row in topo:
            r, rr, cc, up, dn, lf, rt = (int(v) for v in row)
            print(f"  {r:>2}   ({rr},{cc})   {_nbr(up):>5} {_nbr(dn):>5} {_nbr(lf):>5} {_nbr(rt):>5}")
        print("-" * 78, flush=True)

    # ---- Warm-up: JIT-compile every kernel before timing ---------------------
    run_mode("serialized", ctx, F, hb, S, dt, WARMUP_STEPS, record=False)
    run_mode("overlapped", ctx, F, hb, S, dt, WARMUP_STEPS, record=False)

    # ---- Benchmarks ------------------------------------------------------------
    ser = run_mode("serialized", ctx, F, hb, S, dt, STEPS)
    ovl = run_mode("overlapped", ctx, F, hb, S, dt, STEPS)

    if root:
        for name, res in (("serialized", ser), ("overlapped", ovl)):
            print(f"[{name}] Allreduce diagnostics")
            for t in res.trace:
                print(f"  step {t['step']:>4} | E = {_fmt(t['energy'])} | ||u||_L2 = {_fmt(t['l2_norm'])}"
                      f" | ||du/dt||_L2 = {_fmt(t['residual_l2'])}")
        print("-" * 78, flush=True)

    # ---- Parity -----------------------------------------------------------------
    local_diff = np.array([np.max(np.abs(ser.final[1:-1, 1:-1] - ovl.final[1:-1, 1:-1]))], dtype=np.float64)
    gdiff = np.zeros(1, dtype=np.float64)
    ctx.cart.Allreduce(local_diff, gdiff, op=MPI.MAX)
    parity_max_diff = float(gdiff[0])
    parity_passed = parity_max_diff < 1e-12

    # ---- Timings (slowest rank defines the step) ----------------------------------
    tloc = np.array([ser.gpu_ms, ser.wall_ms, ser.mpi_wait_ms,
                     ovl.gpu_ms, ovl.wall_ms, ovl.mpi_wait_ms], dtype=np.float64)
    tmax = np.zeros_like(tloc)
    ctx.cart.Allreduce(tloc, tmax, op=MPI.MAX)
    serialized_ms, ser_wall, ser_wait, overlapped_ms, ovl_wall, ovl_wait = (float(v) for v in tmax)
    speedup = serialized_ms / overlapped_ms if overlapped_ms > 0 else float("nan")

    # ---- PTX audit ------------------------------------------------------------------
    counts, ptx_passed, asm = ptx_audit()

    energies = [t["energy"] for t in ovl.trace]
    energy_monotone = all(energies[k + 1] <= energies[k] for k in range(len(energies) - 1))

    if root:
        print("PTX audit (tiled_stencil_kernel.inspect_asm())")
        for k, v in counts.items():
            print(f"  {k:<16}: {v}")
        print(f"  ptx_audit_passed : {ptx_passed}")
        print("-" * 78)
        print("Benchmark (CUDA events, max over ranks, diagnostics excluded)")
        print(f"  serialized_ms    : {serialized_ms:.3f}   ({serialized_ms / STEPS:.4f} ms/step)  MPI wait {ser_wait:.3f} ms")
        print(f"  overlapped_ms    : {overlapped_ms:.3f}   ({overlapped_ms / STEPS:.4f} ms/step)  MPI wait {ovl_wait:.3f} ms")
        print(f"  wall (MPI.Wtime) : serialized {ser_wall:.3f} ms | overlapped {ovl_wall:.3f} ms")
        print(f"  overlap speedup  : {speedup:.4f}x")
        print("-" * 78)
        print(f"parity_max_diff    : {parity_max_diff:.3e}  -> {'PASS' if parity_passed else 'FAIL'} (< 1e-12)")
        print(f"energy non-increasing over checkpoints: {energy_monotone}")
        print("-" * 78, flush=True)

        with open("tiled_stencil_kernel.ptx", "w") as fh:
            fh.write(asm)

        try:
            with open(os.path.abspath(__file__), "rb") as fh:
                src_sha = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            src_sha = None

        manifest = {
            "schema_version": "1.0",
            "GITHUB_USERNAME": ctx.github_user,
            "gpu_name": ctx.gpu_name,
            "gpu_uuid": ctx.gpu_uuid,
            "compute_capability": f"{ctx.compute_capability[0]}.{ctx.compute_capability[1]}",
            "instance_seed": int(ctx.instance_seed),
            "instance_seed_hex": f"0x{ctx.instance_seed:016x}",
            "mpi_ranks": int(ctx.size),
            "cart_dims": PROC_DIMS,
            "cart_periods": [False, False],
            "rank_topology": [
                {"rank": int(r[0]), "coords": [int(r[1]), int(r[2])],
                 "up": int(r[3]), "down": int(r[4]), "left": int(r[5]), "right": int(r[6])}
                for r in topo
            ],
            "proc_null_value": int(MPI.PROC_NULL),
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [NY_LOCAL + 2, NX_LOCAL + 2],
            "tile_dim": TILE_DIM,
            "shared_tile_dim": SHARED_DIM,
            "steps": STEPS,
            "lambda": LAMBDA_DAMP,
            "dx": DX,
            "dy": DY,
            "kappa_min": kappa_min,
            "kappa_max": kappa_max,
            "dt_max_cfl": dt_max,
            "dt": dt,
            "cfl_safety": CFL_SAFETY,
            "energy_trace_serialized": ser.trace,
            "energy_trace_overlapped": ovl.trace,
            "energy_non_increasing": bool(energy_monotone),
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "serialized_wall_ms": ser_wall,
            "overlapped_wall_ms": ovl_wall,
            "serialized_mpi_wait_ms": ser_wait,
            "overlapped_mpi_wait_ms": ovl_wait,
            "overlap_speedup": speedup,
            "parity_max_diff": parity_max_diff,
            "parity_passed": bool(parity_passed),
            "ptx_counts": counts,
            "ptx_audit_passed": bool(ptx_passed),
            "tasks_completed": [1, 2, 3, 4, 5],
            "solver_source_sha256": src_sha,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open("submission_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
        print("submission_manifest.json written")
        print(json.dumps({k: manifest[k] for k in
                          ("GITHUB_USERNAME", "mpi_ranks", "ptx_audit_passed", "parity_max_diff",
                           "serialized_ms", "overlapped_ms")}, indent=2), flush=True)

    ctx.cart.Barrier()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    try:
        run_solver(sys.argv[1])
    except Exception as exc:  # make sure one failing rank doesn't hang the others
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
