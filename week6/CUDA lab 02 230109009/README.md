# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230109009  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** 7F933171939955F43C23

## Task 1: Warp Divergence Benchmark

N = 1,048,576 elements, 1,000 iterations per element, kernel-only time, average of 10 trials after one warm-up launch.

| Kernel | Avg Kernel Time (ms) | Relative to A |
|---|---|---|
| A (Uniform Path) | 2.168 | 1.00x |
| B (Full Divergence - Interleaved) | 7.158 | 3.30x |
| C (Warp-Aligned Branching) | 3.331 | 1.54x |

Kernel B is the slowest because even and odd threads sit in the same warp and take different branches, so every warp has to run both paths one after the other. Kernel C does the same total work as B, but each warp of 32 threads takes a single path, so there is no divergence inside a warp and it runs about 2.1x faster than B. C is still slower than A because half of its warps run the subtract-divide path, which costs more than multiply-add.

## Task 2: 1D Stencil

Output of `python task2_stencil_1d.py` (N = 100,007):

```
TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08
```

## Task 3: Grid-Stride Scaling

Output of `python task3_grid_stride.py`:

```
Threads launched: 16,384  |  Elements processed: 16,777,216
TASK 3 PASSED: all 16,777,216 elements == 3.5
```

## Task 4: 2D Sobel-X

Output of `python task4_sobel_2d.py` (2048 x 2048 image):

```
TASK 4 PASSED: MAX DELTA = 4.76837158203125e-07
```