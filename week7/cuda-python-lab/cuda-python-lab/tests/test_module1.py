"""Tests for Module 1 (Tasks 1-8). Checks follow the 'Assert' lines in the task sheet."""
import numpy as np
import pytest

from lab_kernels import module1_execution_model as m1

rng = np.random.default_rng(0)


def rand(*shape):
    return rng.random(shape, dtype=np.float32)


@pytest.mark.parametrize("n", [1, 255, 256, 1024, 1000003])
def test_task01_vector_add_1d(n):
    a, b = rand(n), rand(n)
    np.testing.assert_allclose(m1.run_vector_add_1d(a, b), a + b, atol=1e-5)


def test_task02_vector_scale_grid_stride():
    n = 1_000_000                       # far more than 64*256 = 16384 threads
    x = rand(n)
    out = m1.run_vector_scale_grid_stride(x, 2.5, blocks=64, threads=256)
    np.testing.assert_allclose(out, x * np.float32(2.5), atol=1e-5)


def test_task03_matrix_add_2d():
    A, B = rand(513, 1021), rand(513, 1021)
    np.testing.assert_allclose(m1.run_matrix_add_2d(A, B), A + B, atol=1e-5)


@pytest.mark.parametrize("shape", [(17, 33), (300, 129)])
def test_task04_matrix_transpose_naive(shape):
    A = rand(*shape)
    np.testing.assert_allclose(m1.run_matrix_transpose_naive(A), A.T, atol=1e-5)


def test_task05_volume_init_3d():
    D, H, W = 33, 47, 65
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    expected = (z * 10000 + y * 100 + x).astype(np.int32)
    np.testing.assert_equal(m1.run_volume_init_3d(D, H, W), expected)


def test_task06_rgb_to_grayscale():
    img = rng.integers(0, 256, size=(123, 77, 3)).astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    np.testing.assert_allclose(m1.run_rgb_to_grayscale(img),
                               0.299 * R + 0.587 * G + 0.114 * B, atol=1e-3)


def test_task07_vector_lerp():
    n = 100_003
    a, b, t = rand(n), rand(n), rand(n)
    np.testing.assert_allclose(m1.run_vector_lerp(a, b, t), (1 - t) * a + t * b, atol=1e-5)


def test_task08_vector_threshold():
    x = rand(200_001)
    np.testing.assert_equal(m1.run_vector_threshold(x, 0.5), (x >= 0.5).astype(np.uint8))
