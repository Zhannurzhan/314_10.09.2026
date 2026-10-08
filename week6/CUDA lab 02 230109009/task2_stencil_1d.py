"""
Task 2: 1D 3-point smoothing stencil with boundary clamping (halo replication).

    y[i] = 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1]

Run: python task2_stencil_1d.py
"""
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    # Boundary guard: the last block can have threads past the end of the array.
    if idx < N:
        # Left neighbour (clamped at the left edge)
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]

        # Right neighbour (clamped at the right edge)
        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]

        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    """Host wrapper: copy to GPU, launch the kernel, copy the result back."""
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array(N, dtype=np.float32)

    blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK  # round up
    stencil_1d[blocks_per_grid, THREADS_PER_BLOCK](d_in, d_out, N)
    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_stencil(arr):
    """Exact CPU reference using NumPy."""
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 100_007  # odd, non-power-of-two size
    rng = np.random.default_rng(42)
    h_in = rng.random(N, dtype=np.float32)

    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)

    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4), "GPU result does not match CPU reference"
    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    print(f"TASK 2 PASSED: MAX DELTA = {delta}")