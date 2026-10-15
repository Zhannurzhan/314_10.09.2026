"""
Module 1: CUDA Execution Model (Tasks 1-8)

Every kernel here follows the same 3-step recipe:
    1. Work out "which element am I?"   (my thread index)
    2. Check I'm inside the array        (boundary check)
    3. Do my one small piece of work

Each kernel has a matching `run_...` helper below it. The helper does the
boring host-side work: copy data to the GPU, pick grid/block sizes, launch,
copy the result back. The tests call the `run_...` helpers.
"""
import math
import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# Task 01: 1D Vector Addition with Boundary Checks
# ---------------------------------------------------------------------------
@cuda.jit
def vector_add_1d(a, b, out):
    i = cuda.grid(1)                 # global thread number: blockIdx.x*blockDim.x + threadIdx.x
    if i < out.shape[0]:             # extra threads at the end must do nothing
        out[i] = a[i] + b[i]


def run_vector_add_1d(a, b):
    n = a.shape[0]
    d_a, d_b = cuda.to_device(a), cuda.to_device(b)
    d_out = cuda.device_array(n, dtype=np.float32)
    threads = 256
    blocks = (n + threads - 1) // threads    # round UP so every element gets a thread
    vector_add_1d[blocks, threads](d_a, d_b, d_out)
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# Task 02: 1D Grid-Stride Loop Element-wise Scaling
# Fewer threads than elements, so each thread handles several, jumping by
# "stride" = total number of threads in the whole grid.
# ---------------------------------------------------------------------------
@cuda.jit
def vector_scale_grid_stride(x, alpha, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)        # gridDim.x * blockDim.x
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] * alpha


def run_vector_scale_grid_stride(x, alpha, blocks=64, threads=256):
    d_x = cuda.to_device(x)
    d_out = cuda.device_array_like(d_x)
    vector_scale_grid_stride[blocks, threads](d_x, np.float32(alpha), d_out)
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# Task 03: 2D Matrix Element-wise Addition
# cuda.grid(2) returns (x, y). x runs along columns, y along rows.
# ---------------------------------------------------------------------------
@cuda.jit
def matrix_add_2d(A, B, C):
    col, row = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


def _grid_2d(M, N, tile=16):
    """Blocks of tile x tile threads; x covers columns (N), y covers rows (M)."""
    return ((N + tile - 1) // tile, (M + tile - 1) // tile), (tile, tile)


def run_matrix_add_2d(A, B):
    M, N = A.shape
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    grid, block = _grid_2d(M, N)
    matrix_add_2d[grid, block](d_A, d_B, d_C)
    return d_C.copy_to_host()


# ---------------------------------------------------------------------------
# Task 04: 2D Matrix Transpose (naive)
# Reading A[row, col] is fast (neighbouring threads read neighbouring memory).
# Writing B[col, row] is slow (neighbouring threads write far-apart memory).
# Task 13 fixes that with shared memory.
# ---------------------------------------------------------------------------
@cuda.jit
def matrix_transpose_naive(A, B):
    col, row = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]


def run_matrix_transpose_naive(A):
    M, N = A.shape
    d_A = cuda.to_device(A)
    d_B = cuda.device_array((N, M), dtype=A.dtype)
    grid, block = _grid_2d(M, N)
    matrix_transpose_naive[grid, block](d_A, d_B)
    return d_B.copy_to_host()


# ---------------------------------------------------------------------------
# Task 05: 3D Grid Volume Initialization
# Writing vol[z, y, x] on a 3D array; under the hood numba computes the
# flat address z*H*W + y*W + x for us.
# ---------------------------------------------------------------------------
@cuda.jit
def volume_init_3d(vol, D, H, W):
    x, y, z = cuda.grid(3)
    if z < D and y < H and x < W:
        vol[z, y, x] = z * 10000 + y * 100 + x


def run_volume_init_3d(D, H, W):
    d_vol = cuda.device_array((D, H, W), dtype=np.int32)
    block = (8, 8, 8)                                   # 512 threads per block
    grid = ((W + 7) // 8, (H + 7) // 8, (D + 7) // 8)
    volume_init_3d[grid, block](d_vol, D, H, W)
    return d_vol.copy_to_host()


# ---------------------------------------------------------------------------
# Task 06: RGB to Grayscale (ITU-R BT.601)
# One thread per pixel; it reads 3 channels and writes 1 value.
# ---------------------------------------------------------------------------
@cuda.jit
def rgb_to_grayscale(img_rgb, img_gray):
    x, y = cuda.grid(2)
    if y < img_gray.shape[0] and x < img_gray.shape[1]:
        r = img_rgb[y, x, 0]
        g = img_rgb[y, x, 1]
        b = img_rgb[y, x, 2]
        img_gray[y, x] = 0.299 * r + 0.587 * g + 0.114 * b


def run_rgb_to_grayscale(img_rgb):
    H, W, _ = img_rgb.shape
    d_rgb = cuda.to_device(img_rgb)
    d_gray = cuda.device_array((H, W), dtype=np.float32)
    grid, block = _grid_2d(H, W)
    rgb_to_grayscale[grid, block](d_rgb, d_gray)
    return d_gray.copy_to_host()


# ---------------------------------------------------------------------------
# Task 07: Vector LERP
# (1-t)*a + t*b  ==  a + t*(b - a)   -> one subtract + one fused multiply-add
# ---------------------------------------------------------------------------
@cuda.jit
def vector_lerp(a, b, t, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = a[i] + t[i] * (b[i] - a[i])


def run_vector_lerp(a, b, t):
    n = a.shape[0]
    d_out = cuda.device_array(n, dtype=np.float32)
    threads = 256
    blocks = (n + threads - 1) // threads
    vector_lerp[blocks, threads](cuda.to_device(a), cuda.to_device(b),
                                 cuda.to_device(t), d_out)
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# Task 08: Thresholding / Boolean Mask
# No if/else: the comparison itself is 0 or 1, so every thread runs the
# exact same instructions (no branch divergence).
# ---------------------------------------------------------------------------
@cuda.jit
def vector_threshold(x, thresh, mask):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        mask[i] = np.uint8(x[i] >= thresh)


def run_vector_threshold(x, thresh, blocks=128, threads=256):
    d_x = cuda.to_device(x)
    d_mask = cuda.device_array(x.shape[0], dtype=np.uint8)
    vector_threshold[blocks, threads](d_x, np.float32(thresh), d_mask)
    return d_mask.copy_to_host()
