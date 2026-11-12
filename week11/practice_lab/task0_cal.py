"""Phase 0 checkpoint: core count + timer-call latency + VERIFY_HASH[PHASE0]."""
import multiprocessing as mp
import time

from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230109009   
N_CALLS = 1_000_000        # 10^6 back-to-back calls


def many_timer_calls() -> None:
    """Call time.perf_counter_ns() one million times in a row."""
    clock = time.perf_counter_ns          # local name = faster lookup
    for _ in range(N_CALLS):
        clock()


def empty_loop() -> None:
    """Same loop with no call inside, to measure the loop's own cost."""
    for _ in range(N_CALLS):
        pass


if __name__ == "__main__":
    # 1) How many logical cores (threads) does Python see?
    p_log = mp.cpu_count()

    # 2) Time 10^6 calls, 7 times (after 2 warm-up runs).
    #    Total ms for 10^6 calls == average ns per call (same number).
    calls = run_benchmark(many_timer_calls)
    loop = run_benchmark(empty_loop)

    avg_ns = calls.median_ms                      # average latency per call (ns)
    net_ns = calls.median_ms - loop.median_ms     # latency with loop cost removed (ns)

    print(f"P_log (mp.cpu_count())       : {p_log}")
    print(f"10^6 calls  min/median/IQR ms: {calls.min_ms:.4f} / {calls.median_ms:.4f} / {calls.iqr_ms:.4f}")
    print(f"empty loop  min/median/IQR ms: {loop.min_ms:.4f} / {loop.median_ms:.4f} / {loop.iqr_ms:.4f}")
    print(f"Average call latency  (m1)   : {avg_ns:.4f} ns")
    print(f"Net call latency      (m2)   : {net_ns:.4f} ns")

    # 3) The verification line to paste into your report.
    print(emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, net_ns))