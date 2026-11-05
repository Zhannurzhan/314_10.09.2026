# Task 14: Isoefficiency Relation with Gustafson Scaling

## Answer

For a program that follows Gustafson's model, total overhead is `T_o = α(P − 1)`, which grows in a straight line with P. So:

```text
Isoefficiency function:  W = K · α · (P − 1) = Θ(P)
```

This is the best possible isoefficiency function. A Gustafson-compliant program is **ideally scalable** in the isoefficiency sense.

## What isoefficiency means

Isoefficiency asks: **as P grows, how fast must the work `W` grow to keep efficiency constant?** A slower required growth means a more scalable program.

| Symbol | Meaning |
|---|---|
| `W` | Total useful work, measured as time on one processor |
| `T_P` | Parallel run time |
| `T_o` | Total overhead: all processor time that is not useful work |
| `E` | Efficiency |

### Deriving the isoefficiency equation

| # | Statement | Why |
|---:|---|---|
| 1 | `T_o = P·T_P − W` | Total processor time minus useful work |
| 2 | `P·T_P = W + T_o` | Rearrange line 1. |
| 3 | `E = W / (P·T_P) = W / (W + T_o) = 1 / (1 + T_o/W)` | Definition of efficiency, using line 2 |
| 4 | `W = (E / (1 − E)) · T_o` | Solve line 3 for `W`. |
| 5 | `W = K · T_o`, where `K = E/(1 − E)` | `K` is constant when `E` is held constant. **This is the isoefficiency equation.** |

## Proof for Gustafson's model

| # | Statement | Why |
|---:|---|---|
| 1 | `T_P = 1` | Gustafson's normalization (`s + p = 1`) |
| 2 | `W = α + (1 − α)P` | Equivalent serial workload (Task 02) |
| 3 | `T_o = P·1 − (α + (1 − α)P)` | Line 1 of the previous table |
| 4 | `T_o = αP − α = α(P − 1)` | Simplify. |
| 5 | `W = K·α·(P − 1)` | Substitute line 4 into the isoefficiency equation. |
| 6 | `W = Θ(P)` | `K` and `α` are constants. ∎ |

### A check on the efficiency formula

```text
E = W/(W + T_o) = (α + (1 − α)P) / P = (1 − α) + α/P
```

This matches Task 04. ✓

### How Gustafson's growth compares with the exact curve

To hold `E = 1 − α` exactly, set `K = (1 − α)/α`:

```text
W_iso = ((1 − α)/α) · α · (P − 1) = (1 − α)(P − 1)
```

Gustafson's workload can be written as:

```text
W = α + (1 − α)P = 1 + (1 − α)(P − 1) = W_iso + 1
```

So Gustafson's workload is the isoefficiency curve plus 1. This extra 1 is why `E` sits slightly above `1 − α` and approaches it from above.

| P | `W_iso` for `E = 0.96` | Gustafson `W` (`α = 0.04`) |
|---:|---:|---:|
| 16 | 14.40 | 15.40 |
| 1,024 | 982.08 | 983.08 |
| 65,536 | 62,913.60 | 62,914.60 |

## Θ(P) is the best possible

Suppose each processor must do at least one unit of work, so `T_P ≥ 1` unit. Then:

```text
E = W/(P·T_P) ≤ W/P,   so   W ≥ E·P
```

So `W` must grow at least in proportion to P to keep `E` constant. Nothing can have a slower isoefficiency function than `Θ(P)`, and Gustafson's model reaches this.

## A link between the two ideas

**Claim:** when efficiency is held constant, run time stays constant if and only if `T_o = Θ(P)`.

**Proof:**

| # | Statement | Why |
|---:|---|---|
| 1 | `T_P = (W + T_o)/P` | From `T_o = P·T_P − W` |
| 2 | `T_P = (K + 1)·T_o / P` | Using `W = K·T_o`, with `K` constant |
| 3 | `T_P = Θ(1)` if and only if `T_o = Θ(P)` | `K + 1` is a constant in line 2. ∎ |

So **fixed-time scaling is the special case of isoefficiency where overhead grows linearly with P**. If overhead grows faster, you can still hold efficiency constant by growing the work faster, but the run takes longer.

| Program | `T_o` | Isoefficiency | `T_P` at constant `E` | Gustafson-compliant? |
|---|---|---|---|---|
| Gustafson model | `α(P − 1)` | `Θ(P)` | `Θ(1)` | Yes |
| Adding `n` numbers | `2P log P` | `Θ(P log P)` | `Θ(log P)` | No |
| Cannon's matrix multiply | `Θ(P^(3/2))` | `Θ(P^(3/2))` | `Θ(√P)` | No |

Cannon's algorithm is a good example. With `W = N³`, `W = Θ(P^(3/2))` gives `N² = Θ(P)`, so memory per processor stays constant, but time grows like `√P`. This matches the `G(n) = n^(3/2)` case in Task 11.

## What sets the efficiency level

Isoefficiency tells us the growth rate. It does not tell us which efficiency we get. For that, use absolute times. Let `s₀` be the serial time and `w` the parallel work per processor:

```text
W = s₀ + wP
T_P = s₀ + w
T_o = s₀(P − 1)
E → w / (s₀ + w)   as P → ∞
```

So **a larger grain size `w` gives a higher efficiency**. In other words, doing more work on each processor makes `α` smaller. With `s₀ = 1` and `P = 1,024`:

| `w` | `α = s₀/(s₀ + w)` | `E` | Limit |
|---:|---:|---:|---:|
| 1 | 0.50 | 0.5005 | 0.50 |
| 4 | 0.20 | 0.8002 | 0.80 |
| 24 | 0.04 | 0.9600 | 0.96 |
| 99 | 0.01 | 0.9900 | 0.99 |

The constant `K` shows how quickly high efficiency gets expensive:

| Target `E` | 0.50 | 0.80 | 0.90 | 0.96 | 0.99 |
|---|---:|---:|---:|---:|---:|
| `K = E/(1 − E)` | 1 | 4 | 9 | 24 | 99 |

Going from 90% to 99% needs 11 times more work per unit of overhead.

## Critique

**Where the two measures agree:** both say that bigger problems are the way to use bigger machines.

**Where they differ:**

1. **Isoefficiency is more general.** Gustafson assumes one growth rule, `W ∝ P`. Isoefficiency works out the rule needed from the algorithm's real overhead.
2. **Gustafson's overhead is too simple.** In this model `T_o` depends only on P. Real overhead `T_o(W, P)` often depends on `W` as well. That makes the isoefficiency equation harder to solve, and the `Θ(P)` result then no longer holds automatically.
3. **`Θ(P)` is only an ideal in practice.** It holds only while `α` is constant. With a communication term from Task 12, `T_o` gains about `P·C(P)`. For `C(P) = c·log₂P`, this gives `Θ(P log P)`, which is not Gustafson-compliant.
4. **Isoefficiency ignores memory and time limits.** A program may need `W` to grow at a rate that does not fit in memory, or that takes too long to run. Keeping `E` constant does not mean the run is practical.
5. **Isoefficiency hides constants.** Two `Θ(P)` programs can have very different efficiency, as the grain-size table shows.
6. **`W` should come from the best serial algorithm.** With a poor serial baseline, a program can appear more scalable than it is. Also, for large problems `W` is often calculated and not measured (Task 05).

## Conclusion

A Gustafson-compliant program has isoefficiency `Θ(P)`, the best possible. This is only as trustworthy as the assumption that overhead grows linearly with P. Isoefficiency is the better tool for testing that assumption on a real algorithm.

The numbers can be reproduced with `benchmarks/part3_calculations.py`.

## References

1. A. Y. Grama, A. Gupta, V. Kumar, "Isoefficiency: Measuring the Scalability of Parallel Algorithms and Architectures," *IEEE Parallel and Distributed Technology*, vol. 1, no. 3, pp. 12–21, 1993. <https://experts.umn.edu/en/publications/isoefficiency-measuring-the-scalability-of-parallel-algorithms-an/>
2. A. Grama, A. Gupta, G. Karypis, V. Kumar, *Introduction to Parallel Computing*, 2nd ed., Addison Wesley, 2003. Source for the `2P log P` and `Θ(P^(3/2))` examples.
3. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.
