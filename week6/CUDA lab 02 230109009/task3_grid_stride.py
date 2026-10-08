"""
Task 3: Scale a vector of any size with a grid-stride loop.

Only 64 blocks x 256 threads = 16,384 threads are launched, no matter how big
the array is. Each thread walks through the array in steps of 16,384.

Run: python task3_grid_stride.py
"""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64  # fixed on purpose: 64 * 256 = 16,384 threads


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)       # this thread's global index
    stride = cuda.gridsize(1)  # total number of threads in the grid

    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    """Host caller: copy to GPU, scale in place, return the result array."""
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]

    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_arr, np.float32(factor), N)
    cuda.synchronize()

    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 1 << 24  # 16,777,216 elements
    factor = 3.5
    h_arr = np.ones(N, dtype=np.float32)

    result = run_grid_stride(h_arr, factor)

    threads = THREADS_PER_BLOCK * BLOCKS_PER_GRID
    assert result.shape[0] == N
    assert np.all(result == np.float32(factor)), "Not every element was scaled"
    print(f"Threads launched: {threads:,}  |  Elements processed: {N:,}")
    print(f"TASK 3 PASSED: all {N:,} elements == {factor}")