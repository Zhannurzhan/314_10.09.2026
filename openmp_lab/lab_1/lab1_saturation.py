import csv
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from numba import njit

N_SQRT = 50_000_000                                  # longer runs, less noise
TRIALS = 5
P_VALUES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 24, 32, 64]   # adds P=12 to see the knee

@njit(nogil=True)          # nogil lets threads run truly in parallel
def work(n):
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s

def run_team(p: int, reps: int = 1) -> float:
    def task(tid):
        total = 0.0
        for _ in range(reps):
            total += work(N_SQRT)
        return total

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        list(ex.map(task, range(p)))
    return time.perf_counter() - t0

if __name__ == "__main__":
    work(1000)  # JIT warm-up

    # Screenshot mode: python lab1_saturation.py <P> <reps>
    # e.g. python lab1_saturation.py 8 30  -> runs long enough to watch in Task Manager
    if len(sys.argv) == 3:
        p, reps = int(sys.argv[1]), int(sys.argv[2])
        print(f"Running P={p}, reps={reps}. Watch your CPU monitor...")
        print(f"Done in {run_team(p, reps):.2f}s")
        sys.exit()

    TRIALS = 3
    with open("data_saturation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "mean_seconds", "throughput_Msqrt_per_s"])
        for p in P_VALUES:
            times = [run_team(p) for _ in range(TRIALS)]
            mean = sum(times) / len(times)
            thr = p * N_SQRT / mean / 1e6
            w.writerow([p, mean, thr])
            print(f"P={p:3d} | mean = {mean:.3f} s | throughput = {thr:.1f} M sqrt/s")