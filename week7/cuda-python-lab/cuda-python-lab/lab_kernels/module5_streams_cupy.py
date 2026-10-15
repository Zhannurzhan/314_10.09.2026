"""
Module 5: Streams, Concurrency & CuPy Interop (Tasks 35-42)

STUBS: replace each `raise NotImplementedError` with your kernel.
Use Module 1 as the pattern: a @cuda.jit kernel plus a run_... helper.
"""
import math
import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# Task 35: Split work in chunks across 2 streams so copies and kernels overlap.
# ---------------------------------------------------------------------------
def pipeline_dual_stream(h_in, h_out, chunk_size):
    raise NotImplementedError("Task 35 (pipeline_dual_stream) not done yet")


# ---------------------------------------------------------------------------
# Task 36: Batch of matrix @ vector, assigning vector i to stream i % num_streams.
# ---------------------------------------------------------------------------
def multistream_gemv(matrix, vectors, out_vectors, num_streams):
    raise NotImplementedError("Task 36 (multistream_gemv) not done yet")


# ---------------------------------------------------------------------------
# Task 37: CUDA C++ string compiled with cp.RawKernel; sinc(x)=sin(pi x)/(pi x), sinc(0)=1.
# ---------------------------------------------------------------------------
def cupy_rawkernel_sinc(x, out):
    raise NotImplementedError("Task 37 (cupy_rawkernel_sinc) not done yet")


# ---------------------------------------------------------------------------
# Task 38: One C++ source with 3 extern "C" kernels; cp.RawModule(...).get_function(name).
# ---------------------------------------------------------------------------
def cupy_rawmodule_arithmetic(a, b, res_add, res_sub, res_mul):
    raise NotImplementedError("Task 38 (cupy_rawmodule_arithmetic) not done yet")


# ---------------------------------------------------------------------------
# Task 39: Warm up, then time repeats with cuda.event(); return mean and std in ms.
# ---------------------------------------------------------------------------
def cuda_event_profiler(kernel_func, args, num_warmup, num_repeats):
    raise NotImplementedError("Task 39 (cuda_event_profiler) not done yet")


# ---------------------------------------------------------------------------
# Task 40: Capture kernel launches with stream.begin_capture()/end_capture(); launch the graph repeatedly.
# ---------------------------------------------------------------------------
def cuda_graph_capture_exec(a, b, c, num_iterations):
    raise NotImplementedError("Task 40 (cuda_graph_capture_exec) not done yet")


# ---------------------------------------------------------------------------
# Task 41: Managed memory (cuda.managed_array): time first-touch vs prefetched GPU access; check CPU/GPU see same data.
# ---------------------------------------------------------------------------
def benchmark_unified_memory(size_bytes):
    raise NotImplementedError("Task 41 (benchmark_unified_memory) not done yet")


# ---------------------------------------------------------------------------
# Task 42: C++ kernel using cooperative_groups grid.sync() between iterations, cooperative launch.
# ---------------------------------------------------------------------------
def grid_cooperative_barrier(arr, iterations):
    raise NotImplementedError("Task 42 (grid_cooperative_barrier) not done yet")
