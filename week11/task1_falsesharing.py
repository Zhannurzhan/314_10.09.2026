import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
from harness import run_benchmark, emit_verify_tag

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    
    for i in range(iters):
        acc = (acc * mult + (i + 1)) & mask
        arr[slot_idx] = acc  # Direct hardware store to shared L1 cache line
        
    del arr
    shm.close()

def run_parallel_fs(p: int, stride: int, shm_name: str, total_iters: int):
    iters_per_worker = total_iters // p
    processes = []
    
    for t in range(p):
        slot_idx = t * stride
        proc = mp.Process(target=fs_worker, args=(shm_name, slot_idx, iters_per_worker))
        processes.append(proc)
        proc.start()
        
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    M = 24_000_000
    P_LOG = mp.cpu_count()
    
    # Replace this with the Physical Cores value you recorded in Phase 0
    P_PHYS = 8
    
    shm = shared_memory.SharedMemory(create=True, size=4096)
    
    try:
        print(f"--- Table 1A: Stride Sensitivity at p = {P_PHYS} ---")
        strides = [1, 2, 4, 8, 16, 32]
        stride_results = {}
        
        for stride in strides:
            bench_fn = lambda s=stride: run_parallel_fs(P_PHYS, s, shm.name, M)
            stats = run_benchmark(bench_fn, warmup_runs=2, timed_runs=7)
            
            mops = (M / 1e6) / (stats.median_ms / 1000)
            stride_results[stride] = stats.median_ms
            byte_offset = stride * 8
            
            print(f"Stride {stride:2d} ({byte_offset:3d} bytes): Median={stats.median_ms:6.2f}ms, IQR={stats.iqr_ms:5.2f}ms, Throughput={mops:5.2f} Mops/s")
            
        print("\n--- Table 1B: Karp-Flatt Diagnostic Series ---")
        # Generate process scaling sequence up to P_LOG
        p_values = [p for p in [1, 2, 4, 8, 12, 16, 24, 32] if p < P_LOG]
        if P_LOG not in p_values:
            p_values.append(P_LOG)
        
        for s in [1, 16]:
            print(f"\nProcess Scaling Sweep with Stride = {s}")
            T_1 = None
            for p in p_values:
                bench_fn = lambda p_val=p, s_val=s: run_parallel_fs(p_val, s_val, shm.name, M)
                stats = run_benchmark(bench_fn, warmup_runs=2, timed_runs=7)
                
                if p == 1: 
                    T_1 = stats.median_ms
                    
                Sp = T_1 / stats.median_ms
                Ep = Sp / p
                e_p = ((1/Sp) - (1/p)) / (1 - 1/p) if p > 1 else 0.0
                
                print(f"p={p:2d} | T_p={stats.median_ms:7.2f}ms | S_p={Sp:5.2f}x | E_p={Ep:4.2f} | e(p)={e_p:.4f}")
        
        # Emit Phase 1 Verification Hash
        # Student seed embedded based on your established academic ID
        hash_output = emit_verify_tag(
            task_id="PHASE1",
            student_seed=230109009,
            primary_metric=stride_results[1],
            secondary_metric=stride_results[16]
        )
        print(f"\n{hash_output}")
        
    finally:
        shm.close()
        shm.unlink()