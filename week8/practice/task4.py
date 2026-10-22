"""
TASK 4 — IPC Overhead (CPU path)

Pass a float64 array of N = 100,000 / 1,000,000 / 10,000,000 elements
from the parent process to a persistent worker process:

  Path A: multiprocessing Pipe  -> array is PICKLED, pushed through the OS
          pipe byte by byte, and UNPICKLED into a fresh copy in the child.
  Path B: multiprocessing.shared_memory -> the array already lives in a
          shared segment; only its NAME + shape (a few bytes) is sent, and
          the child maps the same physical memory (zero-copy).

Timed = parent sends -> child has a usable array and reads it -> child acks.
The worker is started once, so process start-up is NOT counted.
"""
import time
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

# ---------------------------------------------------------------
STUDENT_ID = 230109009       # <-- PUT YOUR STUDENT ID HERE
SIZES = [100_000, 1_000_000, 10_000_000]
RUNS = 7                       # timed repetitions per N (median reported)
OUTPUT_FILE = "task4_output.txt"
# ---------------------------------------------------------------


def attach_shm(name):
    # Python 3.13+: track=False stops the child's resource tracker from
    # unlinking the segment the parent owns. Older Pythons ignore it.
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def worker(conn):
    """Persistent child process: receives arrays via Path A or Path B."""
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "pipe":                       # Path A: unpickled copy
            arr = msg[1]
            conn.send((arr.size, arr[0], arr[-1]))
        elif kind == "shm":                      # Path B: map shared memory
            _, name, n = msg
            shm = attach_shm(name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            reply = (arr.size, arr[0], arr[-1])
            del arr
            shm.close()
            conn.send(reply)


def time_pipe(conn, arr):
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        conn.send(("pipe", arr))
        size, first, last = conn.recv()
        times.append((time.perf_counter() - t0) * 1000)
        assert size == arr.size and first == arr[0] and last == arr[-1]
    return statistics.median(times)


def time_shm(conn, shm, arr):
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        conn.send(("shm", shm.name, arr.size))
        size, first, last = conn.recv()
        times.append((time.perf_counter() - t0) * 1000)
        assert size == arr.size and first == arr[0] and last == arr[-1]
    return statistics.median(times)


def main():
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=worker, args=(child_conn,))
    proc.start()

    rng = np.random.default_rng(STUDENT_ID)
    rows = []

    # warm-up both paths once so first-call costs aren't measured
    warm = rng.random(1000)
    parent_conn.send(("pipe", warm)); parent_conn.recv()

    for n in SIZES:
        # Data is generated straight into the shared segment (true zero-copy
        # for Path B); Path A sends this very same array through the pipe.
        shm = shared_memory.SharedMemory(create=True, size=n * 8)
        arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
        arr[:] = rng.random(n)

        a_ms = time_pipe(parent_conn, arr)
        b_ms = time_shm(parent_conn, shm, arr)
        rows.append((n, n * 8 / 1024**2, a_ms, b_ms, a_ms / b_ms))

        del arr
        shm.close()
        shm.unlink()

    parent_conn.send(("stop",))
    proc.join()

    line = "=" * 74
    out = [
        line,
        " TASK 4 - IPC Overhead: Pipe Serialization vs Zero-Copy Shared Memory",
        line,
        f" Student ID: {STUDENT_ID} | dtype: float64 | Runs per N (median): {RUNS}",
        "-" * 74,
        f" {'N':>12} | {'Payload':>9} | {'Path A: Pipe':>14} | {'Path B: SHM':>13} | {'A / B':>8}",
        "-" * 74,
    ]
    for n, mb, a, b, ratio in rows:
        out.append(f" {n:>12,} | {mb:>6.2f} MB | {a:>11.3f} ms | {b:>10.3f} ms | {ratio:>7.1f}x")
    out.append(line)
    report = "\n".join(out)
    print(report)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()