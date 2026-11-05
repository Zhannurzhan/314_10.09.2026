# Task 12: Communication Overhead Penalty Integration

## The proposal

The classic law has no separate communication term. I add one, `C(P)`: the communication time each processor spends in one run, measured in units of the base run time.

```text
Rule:  C(1) = 0   (one processor sends no messages)
```

There are two versions, depending on what we hold constant.

| Version | What stays constant | Speedup |
|---|---|---|
| **Model A: time grows** | Work per processor | `S(P) = (α + (1 − α)P) / (1 + C(P))` |
| **Model B: strict fixed time** | Run time | `S(P) = P − α(P − 1) − C(P)·P` |

Model A is the main proposal, because weak-scaling benchmarks usually keep work per processor constant and let the time change. Model B answers the same fixed-time question as Gustafson's original law.

If `C(P) = 0`, both become Gustafson's Law.

## Derivation of Model A

| # | Statement | Why |
|---:|---|---|
| 1 | `T_P = α + (1 − α) + C(P) = 1 + C(P)` | The base run has length 1, and communication adds `C(P)`. |
| 2 | Useful work `= α + (1 − α)P` | Same as Task 02. Sending messages is not useful work. |
| 3 | `T_1 = α + (1 − α)P` | One processor needs no messages, so `C(P)` does not appear. |
| 4 | `S(P) = T_1 / T_P = (α + (1 − α)P) / (1 + C(P))` | Definition of speedup |
| 5 | `E(P) = S(P)/P = ((1 − α) + α/P) / (1 + C(P))` | Divide by P. |

**The new result:** the classic speedup is simply divided by `1 + C(P)`.

## Derivation of Model B

If the run time must stay at 1, communication takes time from computation.

| # | Statement | Why |
|---:|---|---|
| 1 | Parallel compute time `= 1 − α − C(P)` | What remains of the time budget |
| 2 | Useful work `= α + (1 − α − C(P))P` | The P processors work for this shorter time. |
| 3 | `S(P) = α + (1 − α)P − C(P)·P` | Multiply out. The run time is 1. |
| 4 | `S(P) = P − α(P − 1) − C(P)·P` | Using Task 02 |
| 5 | `E(P) = (1 − α − C(P)) + α/P` | Divide by P. |

**Limit:** this only makes sense while `C(P) < 1 − α`. Past that point there is no time left for computation.

## Choosing C(P)

| Form | Where it comes from |
|---|---|
| `C(P) = c · log₂P` | Tree-based collectives such as a global sum on a well-connected network |
| `C(P) = c · (√P − 1)` | For example, long-distance or contended messages on a 2D mesh or torus |
| `C(P) = c · (P − 1)` | All processors take turns on one shared resource |

The `√P − 1` and `P − 1` forms are chosen so that `C(1) = 0`.

A physical formula for one tree-based collective is:

```text
c = (t_s + t_w · m) / T_base
```

where `t_s` is the message start-up time, `t_w` the time per byte, `m` the message size, and `T_base` the base run time. Multiply by the number of collectives in a run.

## Numerical comparison

Using `α = 0.04`, with `C(P) = 0.005·log₂P` and `C(P) = 0.001·(√P − 1)`. These constants are examples only.

| P | Classic | Log `C` | Model A | Model A `E` | Model B | √P `C` | Model A | Model A `E` | Model B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.40 | 0.020 | 15.10 | 0.944 | 15.08 | 0.003 | 15.35 | 0.960 | 15.35 |
| 256 | 245.80 | 0.040 | 236.35 | 0.923 | 235.56 | 0.015 | 242.17 | 0.946 | 241.96 |
| 1,024 | 983.08 | 0.050 | 936.27 | 0.914 | 931.88 | 0.031 | 953.52 | 0.931 | 951.34 |
| 4,096 | 3,932.20 | 0.060 | 3,709.62 | 0.906 | 3,686.44 | 0.063 | 3,699.15 | 0.903 | 3,674.15 |
| 16,384 | 15,728.68 | 0.070 | 14,699.70 | 0.897 | 14,581.80 | 0.127 | 13,956.24 | 0.852 | 13,647.91 |
| 65,536 | 62,914.60 | 0.080 | 58,254.26 | 0.889 | 57,671.72 | 0.255 | 50,131.16 | 0.765 | 46,202.92 |
| 262,144 | 251,658.28 | 0.090 | 230,879.16 | 0.881 | 228,065.32 | 0.511 | 166,550.81 | 0.635 | 117,702.70 |

Three things to notice:

1. **The form of `C(P)` matters more than the constant at large P.** The `√P` model starts better, but it falls behind the log model at about `P = 4,096`.
2. **The classic law is always too high** once `C(P) > 0`.
3. **Models A and B are close while `C(P)` is small.** They separate as it grows.

## What happens as P → ∞

| `C(P)` | Model A speedup | Model A efficiency | Model B |
|---|---|---|---|
| Constant `C₀` | Grows like `(1 − α)P/(1 + C₀)` | `→ (1 − α)/(1 + C₀)` | Grows, if `C₀ < 1 − α` |
| `c·log₂P` | Grows like `(1 − α)P/(c·log₂P)` | `→ 0` slowly | Peaks near `P = 2^190` for these values, far beyond any real machine |
| `c·(√P − 1)` | Grows like `(1 − α)√P/c` | `→ 0` | Peaks, then collapses |
| `c·(P − 1)` | **Stops at `(1 − α)/c`** | `→ 0` | Peaks, then collapses |

Important results:

- **Gustafson's constant efficiency `1 − α` no longer holds** whenever `C(P)` keeps growing.
- **A ceiling returns if `C(P)` grows in proportion to P.** With `c = 0.001`, Model A speedup is 486.0 at `P = 1,024` and 945.6 at `P = 65,536`, and it approaches 960. This is the same kind of limit as Amdahl's (Task 13 has a real example).

### Model B: finding the peak for `C = c(√P − 1)`

```text
S(P) = α + P(1 − α − c(√P − 1))
dS/dP = (1 − α + c) − 1.5c√P = 0
√P = (1 − α + c) / (1.5c)
```

With `α = 0.04` and `c = 0.001`:

- The peak is at `P ≈ 410,454`, where `S ≈ 131,482`.
- At `P = 923,521`, there is no compute time left.

So under a strict time limit, adding processors past the peak makes the machine do **less** work.

## The serial fraction now grows with P

In Model A, the non-parallel share of the run is:

```text
α_eff(P) = (α + C(P)) / (1 + C(P))
```

At `P = 1,024` with the log model, `α_eff = 0.0857`, compared with `α = 0.04`. This explains why `α` measured on a small run can be too low for a large one (Task 07).

## How to use this model on real data

1. Measure `T(P)` at several processor counts, keeping the work per processor constant.
2. Calculate `C(P) = T(P)/T(1) − 1`.
3. Fit `c` for each form using least squares, and keep the form that fits best.

Tasks 22 and 23 collect the data this needs.

## Limits of this model

- All processors are assumed to spend the same communication time, so load imbalance is ignored.
- Messages may overlap with computation in real programs, so the true cost may be less than `C(P)`.
- `C(P)` may depend on the problem size as well as on P. Sun and Ni's `Q_N` term has the same purpose.

The numbers can be reproduced with `benchmarks/part3_calculations.py`.

## References

1. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.
2. X.-H. Sun and L. M. Ni, "Scalable Problems and Memory-Bounded Speedup," *Journal of Parallel and Distributed Computing*, vol. 19, pp. 27–37, 1993.
