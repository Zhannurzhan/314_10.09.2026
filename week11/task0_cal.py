import multiprocessing as mp
import time
from harness import emit_verify_tag

def measure_clock_latency(iterations: int) -> float:
    # Warmup phase to stabilize CPU cache
    for _ in range(1000):
        time.perf_counter_ns()
        
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    
    total_ns = t1 - t0
    return total_ns / iterations

if __name__ == "__main__":
    iterations = 2_000_000
    student_id = 230109009
    
    # Measure average call latency
    avg_latency_ns = measure_clock_latency(iterations)
    
    print(f"Logical Threads (P_log): {mp.cpu_count()}")
    print(f"Average perf_counter_ns latency: {avg_latency_ns:.4f} ns")
    
    # Emit verification hash for Phase 0
    hash_output = emit_verify_tag(
        task_id="PHASE0",
        student_seed=student_id,
        primary_metric=avg_latency_ns,
        secondary_metric=0.0
    )
    print(hash_output)