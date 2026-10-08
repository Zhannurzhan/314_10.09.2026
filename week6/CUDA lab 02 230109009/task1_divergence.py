"""
Task 1: Warp divergence microbenchmark.

Three kernels, each doing 1,000 iterations per element:
  A - every thread runs the same arithmetic (no divergence)
  B - even/odd threads take different paths (divergence inside every warp)
  C - whole warps of 32 threads take the same path (no intra-warp divergence)

Only kernel execution is timed (no host<->device transfers).

Run: python task1_divergence.py
"""
import time
import numpy as np
from numba import cuda

N = 1 << 20  # 1,048,576 elements
ITERS = 1000
THREADS_PER_BLOCK = 256
TRIALS = 10

# float32 constants so the math stays in single precision on the GPU
MUL = np.float32(1.0001)
ADD = np.float32(0.0001)
SUB = np.float32(0.0001)
DIV = np.float32(1.0001)


@cuda.jit
def kernel_a_uniform(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(ITERS):
            v = v * MUL + ADD
        y[idx] = v


@cuda.jit
def kernel_b_interleaved(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:
            # Path 1: multiply-accumulate
            for _ in range(ITERS):
                v = v * MUL + ADD
        else:
            # Path 2: subtract-divide
            for _ in range(ITERS):
                v = (v - SUB) / DIV
        y[idx] = v


@cuda.jit
def kernel_c_warp_aligned(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            # Path 1: multiply-accumulate
            for _ in range(ITERS):
                v = v * MUL + ADD
        else:
            # Path 2: subtract-divide
            for _ in range(ITERS):
                v = (v - SUB) / DIV
        y[idx] = v


def benchmark(kernel, h_y, trials=TRIALS):
    """Return the average kernel-only time in milliseconds."""
    n = h_y.shape[0]
    blocks_per_grid = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Warm-up launch (this is where Numba compiles the kernel)
    d_y = cuda.to_device(h_y)
    kernel[blocks_per_grid, THREADS_PER_BLOCK](d_y, n)
    cuda.synchronize()

    total = 0.0
    for _ in range(trials):
        # Fresh data for every trial; the transfer is NOT timed
        d_y = cuda.to_device(h_y)
        cuda.synchronize()

        t0 = time.perf_counter()
        kernel[blocks_per_grid, THREADS_PER_BLOCK](d_y, n)
        cuda.synchronize()
        total += time.perf_counter() - t0

    return total / trials * 1e3


if __name__ == "__main__":
    h_y = np.ones(N, dtype=np.float32)

    results = [
        ("A (Uniform Path)", benchmark(kernel_a_uniform, h_y)),
        ("B (Full Divergence - Interleaved)", benchmark(kernel_b_interleaved, h_y)),
        ("C (Warp-Aligned Branching)", benchmark(kernel_c_warp_aligned, h_y)),
    ]
    base = results[0][1]

    print(f"N = {N:,} elements, {ITERS} iterations each, average of {TRIALS} trials\n")
    print("| Kernel | Avg Kernel Time (ms) | Relative to A |")
    print("|---|---|---|")
    for name, ms in results:
        print(f"| {name} | {ms:.3f} | {ms / base:.2f}x |")