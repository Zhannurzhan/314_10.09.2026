"""
Module 4: Scientific Computing & Numerical Methods (Tasks 26-34)

STUBS: replace each `raise NotImplementedError` with your kernel.
Use Module 1 as the pattern: a @cuda.jit kernel plus a run_... helper.
"""
import math
import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# Task 26: One Jacobi step of the 2D heat equation on interior points; edges stay fixed.
# ---------------------------------------------------------------------------
def heat_diffusion_2d_step(u_curr, u_next, alpha, dt, dx, dy):
    raise NotImplementedError("Task 26 (heat_diffusion_2d_step) not done yet")


# ---------------------------------------------------------------------------
# Task 27: u_next = 2u - u_prev + C^2 (u[i+1] - 2u[i] + u[i-1]), C = c*dt/dx.
# ---------------------------------------------------------------------------
def wave_equation_1d_step(u_prev, u_curr, u_next, c, dt, dx):
    raise NotImplementedError("Task 27 (wave_equation_1d_step) not done yet")


# ---------------------------------------------------------------------------
# Task 28: One thread per body; loop over all others in shared-memory chunks; update velocity.
# ---------------------------------------------------------------------------
def nbody_all_pairs(positions, velocities, masses, dt, softening):
    raise NotImplementedError("Task 28 (nbody_all_pairs) not done yet")


# ---------------------------------------------------------------------------
# Task 29: Horizontal 1D blur into temp_buf, then vertical 1D blur into img_out.
# ---------------------------------------------------------------------------
def gaussian_blur_2d_separable(img_in, img_out, temp_buf, kernel_1d):
    raise NotImplementedError("Task 29 (gaussian_blur_2d_separable) not done yet")


# ---------------------------------------------------------------------------
# Task 30: 3x3 Sobel Gx, Gy per pixel (clamp edges); magnitude = sqrt(Gx^2+Gy^2), orientation = atan2(Gy, Gx).
# ---------------------------------------------------------------------------
def sobel_edge_detector(img_in, magnitude, orientation):
    raise NotImplementedError("Task 30 (sobel_edge_detector) not done yet")


# ---------------------------------------------------------------------------
# Task 31: CuPy: random x, y in [0,1); pi ~= 4 * mean(x*x + y*y <= 1). Process in batches to save memory.
# ---------------------------------------------------------------------------
def monte_carlo_pi_cupy(num_samples):
    raise NotImplementedError("Task 31 (monte_carlo_pi_cupy) not done yet")


# ---------------------------------------------------------------------------
# Task 32: One thread per point: find nearest centroid by squared distance.
# ---------------------------------------------------------------------------
def kmeans_assign_clusters(points, centroids, labels, distances):
    raise NotImplementedError("Task 32 (kmeans_assign_clusters) not done yet")


# ---------------------------------------------------------------------------
# Task 33: One butterfly stage of radix-2 FFT (input already bit-reversed).
# ---------------------------------------------------------------------------
def fft_radix2_butterfly_stage(data_re, data_im, stage, N):
    raise NotImplementedError("Task 33 (fft_radix2_butterfly_stage) not done yet")


# ---------------------------------------------------------------------------
# Task 34: One thread per pixel; iterate z = z^2 + c until |z| > 2 or max_iter.
# ---------------------------------------------------------------------------
def mandelbrot_escape_time(out, xmin, xmax, ymin, ymax, max_iter):
    raise NotImplementedError("Task 34 (mandelbrot_escape_time) not done yet")
