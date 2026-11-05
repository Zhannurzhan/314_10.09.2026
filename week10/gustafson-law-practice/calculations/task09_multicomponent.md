# Task 09: Multi-Component Workload Partitioning

## Results

| Quantity | `P = 512` | `P = 2,048` |
|---|---:|---:|
| Aggregate serial fraction `α` | 0.08 | 0.08 |
| Scaled speedup `S(P)` | **471.12** | **1,884.24** |
| Efficiency `E(P)` | 92.02% | 92.00% |

## Step 1: Sort the phases

| Phase | Work | Share of parallel run time | Scales with P? |
|---|---|---:|---|
| A | Disk I/O and parameter broadcast | 2% | No, counted as serial |
| B | Boundary message-passing synchronization | 6% | No, counted as serial |
| C | Core domain flux computation | 92% | Yes |

Only phase C is real parallel computation. Phases A and B do not gain from extra processors in this model.

```text
α = A + B = 0.02 + 0.06 = 0.08
1 − α = C = 0.92  ✓
```

**Why B counts as serial:** during boundary exchange, all nodes are active, so it is not serial in the usual sense. But it does not do any of the science. It is overhead, so we group it with the serial part.

## Step 2: Calculate speedup

```text
P = 512:
S = 512 − 0.08 × 511 = 512 − 40.88 = 471.12
Check: 0.08 + 0.92 × 512 = 471.12  ✓
E = 471.12 / 512 = 0.920156

P = 2048:
S = 2048 − 0.08 × 2047 = 2048 − 163.76 = 1884.24
Check: 0.08 + 0.92 × 2048 = 1884.24  ✓
E = 1884.24 / 2048 = 0.920039
```

## Step 3: Which phase costs the most?

The speedup lost to each phase is its share times `(P − 1)`.

| | `P = 512` | `P = 2,048` |
|---|---:|---:|
| Lost to A | 10.22 | 40.94 |
| Lost to B | 30.66 | 122.82 |
| **Total lost** | **40.88** | **163.76** |

Phase B causes **three quarters** of the loss at any P, because `0.06 / 0.08 = 75%`. So it is the first place to look for improvement.

## Step 4: What if a phase is improved?

Here we assume the run time stays constant, so any time saved goes to more computation in phase C.

| Change | `α` | `S(512)` | `S(2048)` |
|---|---:|---:|---:|
| As measured | 0.08 | 471.12 | 1,884.24 |
| Remove A | 0.06 | 481.34 | 1,925.18 |
| Halve B | 0.05 | 486.45 | 1,945.65 |
| Halve both A and B | 0.04 | 491.56 | 1,966.12 |
| Remove B | 0.02 | 501.78 | 2,007.06 |

Halving B gains more than removing A completely.

## Warnings about this model

1. **A small detail about phase B.** If no processor does useful work during A or B, the exact speedup is `S = 0.92P`, which gives 471.04 and 1,884.16. This is only `α = 0.08` less than the results above, so the choice makes almost no difference.

2. **The percentages might not stay constant.** The task gives one set of percentages and we used it for both values of P. In a real program:
   - **I/O:** if the input data grows with P and is read through one path, this time grows.
   - **Broadcast:** the time usually grows like `log P`.
   - **Boundary exchange:** with nearest-neighbour messages and a fixed block per node, this time can stay nearly constant. Network contention can still make it grow.

   So the values at `P = 2,048` may be optimistic, unless the percentages were measured at that size. Task 12 models this.

The numbers can be reproduced with `benchmarks/part2_calculations.py`.
