"""
Task 4: 2D Sobel horizontal filter (Sobel-X).

    Kx = [[-1, 0, +1],
          [-2, 0, +2],
          [-1, 0, +1]]

Interior pixels get the convolution result, border pixels are set to 0.

Run: python task4_sobel_2d.py
"""
import math
import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    # Stay inside the image: the grid can be slightly larger than the image.
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            # Internal pixel
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1]
            )
        else:
            # Outer border pixel
            d_out[row, col] = 0.0


def run_sobel(h_img):
    """Host wrapper: copy image to GPU, run Sobel-X, return the filtered matrix."""
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)

    # Grid geometry computed from the image size (rounded up).
    # cuda.grid(2) returns (col, row), so the first grid dimension covers columns.
    blocks_x = math.ceil(cols / THREADS_2D[0])
    blocks_y = math.ceil(rows / THREADS_2D[1])

    sobel_x_kernel[(blocks_x, blocks_y), THREADS_2D](d_in, d_out, rows, cols)
    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_sobel(img):
    """CPU reference with NumPy slicing (used only for the self-test)."""
    out = np.zeros_like(img, dtype=np.float32)
    out[1:-1, 1:-1] = (
        -1.0 * img[:-2, :-2] + 1.0 * img[:-2, 2:]
        - 2.0 * img[1:-1, :-2] + 2.0 * img[1:-1, 2:]
        - 1.0 * img[2:, :-2] + 1.0 * img[2:, 2:]
    )
    return out


if __name__ == "__main__":
    rows, cols = 2048, 2048
    rng = np.random.default_rng(42)
    h_img = rng.random((rows, cols), dtype=np.float32)

    h_out = run_sobel(h_img)
    ref = cpu_sobel(h_img)

    assert h_out.shape == (rows, cols)
    assert np.allclose(h_out, ref, atol=1e-4), "GPU result does not match CPU reference"
    assert np.all(h_out[0, :] == 0.0) and np.all(h_out[-1, :] == 0.0), "Border rows not zeroed"
    assert np.all(h_out[:, 0] == 0.0) and np.all(h_out[:, -1] == 0.0), "Border columns not zeroed"
    delta = float(np.max(np.abs(h_out - ref)))
    print(f"TASK 4 PASSED: MAX DELTA = {delta}")