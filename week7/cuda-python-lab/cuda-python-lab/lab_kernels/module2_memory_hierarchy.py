"""
Module 2: Memory Hierarchy & Shared Memory (Tasks 9-17)

STUBS: replace each `raise NotImplementedError` with your kernel.
Use Module 1 as the pattern: a @cuda.jit kernel plus a run_... helper.
"""
import math
import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# Task 09: Copy with stride 1 vs stride S; time with cuda.event() for S=1,2,4,8,16,32.
# ---------------------------------------------------------------------------
def benchmark_strided_access(in_arr, out_arr, stride):
    raise NotImplementedError("Task 09 (benchmark_strided_access) not done yet")


# ---------------------------------------------------------------------------
# Task 10: Moving average, radius R. Shared tile of 256+2R, load halo with clamping, syncthreads, then average.
# ---------------------------------------------------------------------------
def stencil_1d_shared(in_arr, out_arr, radius):
    raise NotImplementedError("Task 10 (stencil_1d_shared) not done yet")


# ---------------------------------------------------------------------------
# Task 11: 1D convolution, same as np.convolve(x, kernel, 'same'). Load tile + halos into shared memory.
# ---------------------------------------------------------------------------
def conv_1d_shared_tiled(x, kernel, y):
    raise NotImplementedError("Task 11 (conv_1d_shared_tiled) not done yet")


# ---------------------------------------------------------------------------
# Task 12: C = A @ B using 16x16 shared tiles sA, sB. Loop over tiles, syncthreads before and after the inner sum.
# ---------------------------------------------------------------------------
def matmul_tiled_shared(A, B, C):
    raise NotImplementedError("Task 12 (matmul_tiled_shared) not done yet")


# ---------------------------------------------------------------------------
# Task 13: Transpose via cuda.shared.array((32, 33)) -- the +1 column removes bank conflicts.
# ---------------------------------------------------------------------------
def transpose_bank_conflict_free(A, B):
    raise NotImplementedError("Task 13 (transpose_bank_conflict_free) not done yet")


# ---------------------------------------------------------------------------
# Task 14: Each block sums its chunk in shared memory (stride halves each step) -> block_sums[blockIdx.x].
# ---------------------------------------------------------------------------
def reduction_sum_shared(in_arr, block_sums):
    raise NotImplementedError("Task 14 (reduction_sum_shared) not done yet")


# ---------------------------------------------------------------------------
# Task 15: Like Task 14 but keep max value AND its index; on a tie keep the smaller index.
# ---------------------------------------------------------------------------
def reduction_argmax_shared(in_arr, block_max_vals, block_max_indices):
    raise NotImplementedError("Task 15 (reduction_argmax_shared) not done yet")


# ---------------------------------------------------------------------------
# Task 16: 64-tap FIR filter; weights in cuda.const.array_like(weights) inside the kernel.
# ---------------------------------------------------------------------------
def constant_memory_filter(signal, out):
    raise NotImplementedError("Task 16 (constant_memory_filter) not done yet")


# ---------------------------------------------------------------------------
# Task 17: Time host->device copies from a normal numpy array vs cuda.pinned_array. Return bandwidths in GB/s.
# ---------------------------------------------------------------------------
def benchmark_pinned_transfer(size_bytes):
    raise NotImplementedError("Task 17 (benchmark_pinned_transfer) not done yet")
