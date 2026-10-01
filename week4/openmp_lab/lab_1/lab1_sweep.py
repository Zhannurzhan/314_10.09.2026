import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def run_team(p: int) -> float:
    # Barrier forces all p threads to be alive at the same time.
    # Without it, the pool may reuse one idle thread for many tasks
    # and you would not actually be creating p threads.
    barrier = threading.Barrier(p)

    def task(tid):
        barrier.wait()
        return tid

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        futures = [ex.submit(task, tid) for tid in range(p)]
        for f in futures:
            f.result()          # join
    return time.perf_counter() - t0

if __name__ == "__main__":
    TRIALS = 5
    P_VALUES = [1, 2, 4, 8, 16, 32, 64]

    run_team(4)  # warm-up

    with open("data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "mean_seconds"])
        for p in P_VALUES:
            times = [run_team(p) for _ in range(TRIALS)]
            mean = sum(times) / len(times)
            w.writerow([p, mean])
            print(f"P={p:3d} | mean = {mean*1000:.3f} ms")