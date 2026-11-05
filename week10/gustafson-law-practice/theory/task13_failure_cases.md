# Task 13: Boundary Case: When Gustafson Fails

## Short answer

Gustafson's Law needs two things to hold as the problem grows:

1. Work must grow only in proportion to P, so run time stays constant.
2. Serial time must stay about the same.

Each case below breaks one of these.

| Case | Assumption broken | Run time as P grows | Efficiency as P grows |
|---|---|---|---|
| **A. Conjugate gradient on a 3D grid** | Work grows faster than P | Grows at least like `P^(1/3)` | `→ 0` |
| **B. Task farm with a central locked queue** | Serial time grows with P | Grows like P after a limit | `→ 0` |

Gustafson's Law predicts that run time stays constant and efficiency stays near `1 − α`. Both predictions fail in these cases.

## Case A: Conjugate gradient (CG) on a 3D grid

**What it does:** CG solves large sparse systems of equations, such as the 3D Poisson equation, through repeated iterations. Each iteration does some local work, swaps boundary values with neighbours and computes global dot products.

**Type of failure:** making the problem bigger changes the algorithm's time complexity.

### The analysis

Each processor holds a fixed block, so the total grid has `n = N³` unknowns with `n ∝ P`.

| # | Statement | Why |
|---:|---|---|
| 1 | `N ∝ P^(1/3)` | Total unknowns `N³` grow with P. |
| 2 | Condition number `κ ∝ N²` | Standard result for these grids: `κ` is of order `n^(2/d)`, with `d = 3`. |
| 3 | Iterations `k ∝ √κ ∝ N ∝ P^(1/3)` | CG needs about `√κ` iterations. |
| 4 | Per-iteration time `= t_local + t_halo + 2·t_r·log₂P` | Local work and neighbour swaps stay fixed. The two dot products per iteration each need a global sum. |
| 5 | `T(P) ∝ P^(1/3) · (t + 2·t_r·log₂P)` | Line 3 times line 4 |

Line 3 alone breaks the law, because **a bigger problem needs more iterations**. Even with free communication (`t_r = 0`), the time still grows like `P^(1/3)`, and:

```text
Time-based efficiency = T(1)/T(P) = P^(−1/3) → 0
```

This also fits Task 11, because total work grows as `n^(4/3)`, so `G(P) = P^(4/3)` rather than `P`.

### Numbers

These use the example values `t = 1` and `t_r = 0.01`.

| P | Iterations (factor) | Per-iteration time | Total time | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00 | 1.00 | 1.000 |
| 8 | 2 | 1.06 | 2.12 | 0.472 |
| 64 | 4 | 1.12 | 4.48 | 0.223 |
| 512 | 8 | 1.18 | 9.44 | 0.106 |
| 4,096 | 16 | 1.24 | 19.84 | 0.050 |
| 32,768 | 32 | 1.30 | 41.60 | 0.024 |
| 262,144 | 64 | 1.36 | 87.04 | 0.011 |

### The trap: per-iteration results can hide this

The per-iteration time grows only from 1.00 to 1.36. A benchmark that reports **time per iteration** would look like good weak scaling, but the **time to solution** grows 87 times. Always check which one a weak-scaling result reports.

### What helps

A good preconditioner such as multigrid can keep the iteration count nearly constant. That restores most of the weak scaling, though its coarse levels add communication of their own.

## Case B: Task farm with a central locked queue

**What it does:** all tasks sit in one shared queue protected by one lock. Each worker takes a task, works on it, and returns for more. This is a common design for simple thread pools.

**Type of failure:** synchronization contention.

### The analysis

Each worker does `k` tasks, so total tasks grow with P, as weak scaling requires. Each task needs:

- `w` = time doing the task, which can run in parallel
- `q` = time holding the lock, during which nobody else can use the queue

| # | Statement | Why |
|---:|---|---|
| 1 | Time for one worker alone `= k(w + q)` | The base case |
| 2 | Total queue time needed `= P·k·q` | Only one worker can hold the lock, so these times add up. |
| 3 | `T(P) = k · max(w + q, P·q)` | The run cannot finish before each worker's own tasks are done, or before the queue finishes its total. |
| 4 | Limit point: `P_sat = (w + q)/q` | Where the two values in line 3 are equal |
| 5 | `S(P) = P·k(w + q) / T(P) = min(P, (w + q)/q)` | Total work divided by time |

**Serial time grows in proportion to P.** Task 03 warned about this case: if serial work grows as fast as parallel work, making the problem bigger does not help. Here, `(w + q)/q` is exactly Amdahl's ceiling `1/f` with `f = q/(w + q)`. **So Amdahl's ceiling returns, even though the problem is scaling up.**

### Numbers

These use the example values `w = 1,000 µs` and `q = 10 µs`, so the lock is held for only 1% of each task and `P_sat = 101`.

| P | Time per `k` tasks (µs) | Speedup | Efficiency | Queue busy |
|---:|---:|---:|---:|---:|
| 1 | 1,010 | 1.0 | 1.000 | 1.0% |
| 16 | 1,010 | 16.0 | 1.000 | 15.8% |
| 64 | 1,010 | 64.0 | 1.000 | 63.4% |
| 101 | 1,010 | 101.0 | 1.000 | 100% |
| 256 | 2,560 | 101.0 | 0.395 | 100% |
| 1,024 | 10,240 | 101.0 | 0.099 | 100% |
| 4,096 | 40,960 | 101.0 | 0.025 | 100% |

Up to the limit, scaling looks perfect. A test on 64 cores would show nothing wrong. Past the limit, run time grows in a straight line with P: at `P = 1,024`, it is 10.14 times longer.

**This table is the best case.** On real hardware, contended locks also cause cache-line transfers between cores, so performance often gets worse near the limit and may even fall as threads are added.

### What helps

Remove the shared point of contention. Examples are per-worker queues with work stealing, taking tasks in batches, or splitting one lock into many.

## Why the constant-time premise breaks

Run time stays constant only when all parts of the run stay constant per processor. In a general form:

```text
T(P) = serial(P) + parallel_work(P)/P + communication(P)
```

| Case | Which term grows | Why |
|---|---|---|
| A | `parallel_work(P)/P` | Iterations grow with problem size. |
| A | `communication(P)` | Global sums grow with `log P`. |
| B | `serial(P)` | Lock time adds up across all workers. |

Gustafson's formula uses a single constant `α`, so it cannot show any of these.

## Other cases to look at

| Algorithm | Why weak scaling fails | See |
|---|---|---|
| Dense matrix multiply or LU at full memory | Work grows like `P^(3/2)`, so time grows like `√P` | Tasks 11, 16 |
| Parallel 3D FFT | Needs all-to-all exchanges, which are limited by network bandwidth | |
| BFS on power-law graphs | Irregular work and data-dependent synchronization | Task 20 |

## How to test for these failures

1. Measure **time to solution**, not only time per step.
2. Test beyond the core counts you plan to use if you can, because Case B looks perfect until its limit.
3. Work out `G(P)` for your algorithm (Task 11) before you trust a weak-scaling prediction.
4. Measure time spent in locks and communication at each P.

The numbers can be reproduced with `benchmarks/part3_calculations.py`. The values of `t`, `t_r`, `w` and `q` are examples. The growth rates do not depend on them.

## References

1. J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, Carnegie Mellon University, 1994. <https://ece.uwaterloo.ca/~ece602/MISC/CGM.pdf>
2. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.
3. X.-H. Sun and L. M. Ni, "Scalable Problems and Memory-Bounded Speedup," *Journal of Parallel and Distributed Computing*, vol. 19, pp. 27–37, 1993.
