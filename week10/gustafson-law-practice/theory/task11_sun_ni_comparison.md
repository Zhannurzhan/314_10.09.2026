# Task 11: The Sun and Ni Memory-Bounded Speedup Comparison

## Short answer

Sun and Ni asked what happens when a problem grows until it **fills the memory**. Their formula has one extra function, `G(n)`, which says how much the work grows when memory grows `n` times.

- `G(n) = 1` gives Amdahl's Law.
- `G(n) = n` gives Gustafson's Law.
- `G(n) > n` gives a speedup above Gustafson's, but the run takes longer.
- `G(n) < n` gives a speedup below Gustafson's.

So Gustafson's Law is one special case, and it is accurate only when work grows at the same rate as memory.

## The three models

Take a base problem on one processor. Let `f` be its serial share, so `1 − f` is its parallel share. With `n` processors:

| Model | Limited by | Work grows by | Speedup |
|---|---|---|---|
| Amdahl (fixed-size) | The job itself | `G(n) = 1` | `1 / (f + (1 − f)/n)` |
| Gustafson (fixed-time) | Run time | `G(n) = n` | `f + (1 − f)·n` |
| Sun-Ni (memory-bounded) | Memory | Any `G(n)` | See below |

### Derivation of the Sun-Ni formula

| # | Statement | Why |
|---:|---|---|
| 1 | Work on `n` processors `= f + (1 − f)·G(n)` | The serial part stays the same. The parallel part grows by `G(n)`. |
| 2 | One-processor time `= f + (1 − f)·G(n)` | One processor must do all of this work. |
| 3 | `n`-processor time `= f + (1 − f)·G(n)/n` | The parallel part is shared among `n` processors. |
| 4 | `S(n) = (f + (1 − f)·G(n)) / (f + (1 − f)·G(n)/n)` | Divide line 2 by line 3. |

Sun and Ni's paper writes this as `(W₁ + G(N)·W_N) / (W₁ + G(N)·W_N/N)`, and adds a communication term `Q_N` to the bottom when communication is included.

### Both older laws are special cases

```text
G(n) = 1:  S = (f + 1 − f) / (f + (1 − f)/n)      = 1 / (f + (1 − f)/n)      Amdahl
G(n) = n:  S = (f + (1 − f)n) / (f + 1 − f)       = f + (1 − f)n             Gustafson
```

With `G(n) = n` the run time equals 1 for every `n`. So here `f` is also the serial share of the parallel run, which is Gustafson's `α`.

## How to find G(n)

Link work and memory through the problem size `N`, then let memory grow `n` times.

| Problem | Memory | Work | `N` grows by | `G(n)` |
|---|---|---|---:|---:|
| Fixed input data | Fixed | Fixed | 1 | `1` |
| Grid, one sweep | `N³` | `N³` | `n^(1/3)` | `n` |
| Grid, steps grow with `N` (Task 10) | `N³` | `N⁴` | `n^(1/3)` | `n^(4/3)` |
| Dense matrix multiply | `N²` | `N³` | `n^(1/2)` | `n^(3/2)` |

Sun and Ni give `G(N) = N^(3/2)` for matrix multiplication.

## Comparison in numbers

Using `f = 0.04`. "Time" is run time compared with the base run.

| n | `G = 1` | Time | `G = n` | Time | `G = n^(3/2)` | Time | `G = √n` | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 10.00 | 0.100 | 15.40 | 1 | 15.85 | 3.88 | 13.86 | 0.280 |
| 64 | 18.18 | 0.055 | 61.48 | 1 | 63.67 | 7.72 | 48.25 | 0.160 |
| 256 | 22.86 | 0.044 | 245.80 | 1 | 255.34 | 15.40 | 154.00 | 0.100 |
| 1,024 | 24.43 | 0.041 | 983.08 | 1 | 1,022.67 | 30.76 | 439.43 | 0.070 |
| 4,096 | 24.85 | 0.040 | 3,932.20 | 1 | 4,093.34 | 61.48 | 1,117.82 | 0.055 |
| 16,384 | 24.96 | 0.040 | 15,728.68 | 1 | 16,378.67 | 122.92 | 2,587.79 | 0.048 |

### A general rule

Let `x = (1 − f)·G(n)`, so `S = (f + x) / (f + x/n)`. Then:

```text
dS/dx = f(1 − 1/n) / (f + x/n)²  > 0     (for f > 0 and n > 1)
```

So **speedup always rises as `G(n)` rises**. As `x → ∞`, `S → n`, so it never reaches `n`. This means, for `G(n) ≥ n`:

```text
Amdahl ≤ Gustafson ≤ Sun-Ni < n
```

## When Gustafson's Law underestimates

**Condition: work grows faster than memory (`G(n) > n`), and you accept a longer run.**

Dense matrix multiply on `n = 1,024` with `f = 0.04`:

| | Gustafson (fixed time) | Sun-Ni (fill memory) |
|---|---:|---:|
| Speedup | 983.08 | 1,022.67 |
| Work completed | 983.08 | 31,457.32 |
| Run time | 1 | 30.76 |
| Growth of `N` | 10.08 times | 32 times |

Two things are underestimated here:

1. **Speedup, by a little.** Since `S < n` and Gustafson's value is at least `(1 − f)n`, the error is below `1/(1 − f)`, which is about 4.2% here. It is small because the serial share becomes tiny: the serial part is only 0.13% of this longer run.
2. **Problem size, by a lot.** The largest problem you can solve is 32 times bigger in work, if you can wait about 31 times longer.

Under a strict fixed-time rule, the matrix side grows only 10.08 times, and about 90% of each node's memory stays unused. Sun and Ni note that when `G(N) > N`, execution time is likely to be the real limit.

## When Gustafson's Law overestimates

**Condition: work cannot grow as fast as the processor count (`G(n) < n`).**

With `G(n) = √n` at `n = 1,024`, speedup is 439.43, not 983.08. This is 55% lower. Common causes:

1. **The input is fixed.** If the data will not grow, `G(n) = 1` and Amdahl applies.
2. **Memory does not grow with processors.** In a shared-memory computer, adding threads adds no memory. Once memory is full, `G` stops growing and the speedup approaches Amdahl's ceiling. This matters for the benchmark in Task 22.
3. **Extra data needs little extra work.** With an indexed lookup, for example, more data barely adds work per query.
4. **Serial or communication time grows.** The Sun-Ni term `Q_N` and Task 12 cover this. Neither `G(n)` nor Gustafson's basic formula helps here.

## Summary

| Situation | `G(n)` | Gustafson's Law |
|---|---|---|
| Job and data truly fixed | `1` | Greatly overestimates |
| Work grows slower than memory | `< n` | Overestimates |
| Work and memory grow together | `n` | Correct |
| Work grows faster than memory, longer runs allowed | `> n` | Slightly underestimates speedup, greatly underestimates problem size |
| Work grows faster than memory, time is fixed | Limited to `n` | Correct, with memory left unused |

**In practice:** work out `G(n)` for your algorithm first. Then choose the law that matches your real limit, whether that is the job, the time or the memory.

The numbers can be reproduced with `benchmarks/part3_calculations.py`.

## References

1. X.-H. Sun and L. M. Ni, "Scalable Problems and Memory-Bounded Speedup," *Journal of Parallel and Distributed Computing*, vol. 19, pp. 27–37, 1993. <https://ntrs.nasa.gov/citations/19930007365>
2. X.-H. Sun and L. M. Ni, "Another View on Parallel Speedup," *Proceedings of Supercomputing '90*.
3. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.
