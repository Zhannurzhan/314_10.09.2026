import multiprocessing as mp
import math
import time
from harness import run_benchmark, emit_verify_tag

def stage_a(n_elements: int) -> float:
    """Stage A: Strictly Serial Reduction (12 transcendental ops per element)"""
    acc = 0.0
    for i in range(n_elements):
        acc += math.sin(i) + math.sqrt(i)
        acc = math.sin(acc) + math.sqrt(abs(acc))
        acc = math.sin(acc) + math.sqrt(abs(acc))
    return acc

def stage_b_worker(chunk_size: int) -> float:
    """Stage B: Embarrassingly Parallel Compute (35 FLOPs/trig per element)"""
    acc = 0.0
    for i in range(chunk_size):
        v = i * 0.1
        for _ in range(7):
            v = v * 1.5 + math.sin(v)
        acc += v
    return acc

def execute_pipeline(pool, p: int, N: int):
    # Stage A: First 5% of elements executed in parent
    n_a = int(0.05 * N)
    stage_a(n_a)
    
    # Stage B: Remaining elements partitioned evenly across pool
    base_chunk = N // p
    chunks = [base_chunk] * p
    chunks[-1] += N % p  # add any remainder to the last worker
    
    pool.map(stage_b_worker, chunks)

def measure_stage_a(N: int, timed_runs: int = 5) -> float:
    n_a = int(0.05 * N)
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        stage_a(n_a)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    samples.sort()
    return samples[timed_runs // 2]

if __name__ == "__main__":
    P_PHYS = 8
    P_LOG = 12
    N_STRONG = 1_600_000
    N0_WEAK = 250_000
    
    print(f"--- CPU Detected: P_phys={P_PHYS}, P_log={P_LOG} ---")
    
    # --- Experiment 2A: Strong Scaling Sweep ---
    print("\n--- Experiment 2A: Strong Scaling (N = 1.6M) ---")
    T_1_strong = None
    strong_results = {}
    
    for p in range(1, P_LOG + 1):
        with mp.Pool(processes=p) as pool:
            bench_fn = lambda pool=pool, p=p: execute_pipeline(pool, p, N_STRONG)
            stats = run_benchmark(bench_fn, warmup_runs=1, timed_runs=5)
            T_p = stats.median_ms
            
            if p == 1:
                T_1_strong = T_p
            
            S_p = T_1_strong / T_p
            E_p = S_p / p
            strong_results[p] = S_p
            print(f"p={p:2d} | T_p={T_p:7.2f} ms | S_p={S_p:5.2f}x | E_p={E_p:4.2f}")
            
    # --- Experiment 2B: Weak Scaling Sweep ---
    print("\n--- Experiment 2B: Weak Scaling (N_0 = 250k/worker) ---")
    T_1_weak = None
    gustafson_results = {}
    
    for p in range(1, P_LOG + 1):
        N_weak = p * N0_WEAK
        with mp.Pool(processes=p) as pool:
            bench_fn = lambda pool=pool, p=p, N=N_weak: execute_pipeline(pool, p, N)
            stats = run_benchmark(bench_fn, warmup_runs=1, timed_runs=5)
            T_p = stats.median_ms
            
            if p == 1:
                T_1_weak = T_p
                
            T_stage_a = measure_stage_a(N_weak)
            alpha = T_stage_a / T_p
            E_p = T_1_weak / T_p
            S_Gustafson = p - alpha * (p - 1)
            gustafson_results[p] = S_Gustafson
            
            print(f"p={p:2d} | T_p={T_p:7.2f} ms | alpha={alpha:.4f} | E_p={E_p:4.2f} | S_Gustaf={S_Gustafson:5.2f}x")
            
    # --- Analytical Deliverables ---
    print("\n--- Analytical Ceilings ---")
    T_stage_a_strong_1 = measure_stage_a(N_STRONG)
    s_fraction = T_stage_a_strong_1 / T_1_strong
    S_inf = 1.0 / s_fraction if s_fraction > 0 else float('inf')
    print(f"1. Serial Fraction (s) at p=1: {s_fraction:.4f}")
    print(f"   Amdahl's Max Theoretical Speedup (S_inf): {S_inf:.2f}x")
    print(f"   Observed Strong Speedup at p={P_LOG}: {strong_results[P_LOG]:.2f}x")
    
    # Phase 2 Verification Hash (using 230109009 student seed)
    hash_output = emit_verify_tag(
        task_id="PHASE2",
        student_seed=230109009,
        primary_metric=strong_results[P_LOG],
        secondary_metric=gustafson_results[P_LOG]
    )
    print(f"\n{hash_output}")