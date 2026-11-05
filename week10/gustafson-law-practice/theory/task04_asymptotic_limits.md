# Task 04: Asymptotic Analysis of Parallel Efficiency

## Results

| | Gustafson | Amdahl |
|---|---|---|
| Efficiency `E(P)` | `(1 − α) + α/P` | `1 / (sP + 1 − s)` |
| Limit as `P → ∞` | **`1 − α`** | **`0`** (when `s > 0`) |
| Speedup as `P → ∞` | Grows without limit | Stops at `1/s` |

## Part A: Gustafson's limit

### Step 1: Simplify

```text
E(P) = S(P)/P
     = (α + (1 − α)P) / P
     = (1 − α) + α/P
```

### Step 2: Take the limit

The value `α` is a constant, so `α/P → 0` as `P → ∞`. Therefore:

```text
lim E(P) = 1 − α    as P → ∞
```

### Step 3: Formal proof

Take any `ε > 0`. We need a point after which `E(P)` stays within `ε` of `1 − α`.

```text
|E(P) − (1 − α)| = α/P
```

If `α = 0`, this is always 0, so the claim is true. If `α > 0`, then `α/P < ε` for every `P > α/ε`. So the limit is `1 − α`. ∎

### Step 4: E(P) falls steadily to this limit

For `α > 0`:

```text
E(P + 1) − E(P) = α(1/(P + 1) − 1/P) < 0
```

So `E(P)` always falls as P rises. It also stays above the limit, since `E(P) − (1 − α) = α/P > 0`. This means **`1 − α` is a guaranteed minimum efficiency**, as long as the model's assumptions hold.

## Part B: Amdahl's limit

### Step 1: Simplify

```text
E(P) = S(P)/P
     = 1 / (P · (s + (1 − s)/P))
     = 1 / (sP + 1 − s)
```

### Step 2: Formal proof that the limit is 0 when `s > 0`

Since `1 − s ≥ 0`, the bottom is at least `sP`, so:

```text
0 < E(P) ≤ 1/(sP)
```

Take any `ε > 0`. For every `P > 1/(sε)`, we get `1/(sP) < ε`, so `E(P) < ε`. So the limit is 0. ∎

Efficiency falls like `1/(sP)`. Each time P doubles, efficiency roughly halves, once P is large.

### Step 3: Where efficiency drops to 50%

```text
1 / (sP + 1 − s) = 1/2   →   P = 1/s + 1
```

With `s = 0.05`, efficiency has fallen to 50% at only `P = 21`.

## Part C: Comparison

Using `s = α = 0.05`:

| P | Gustafson `E` | Amdahl `E` |
|---:|---:|---:|
| 1 | 1.000000 | 1.000000 |
| 16 | 0.953125 | 0.571429 |
| 256 | 0.950195 | 0.072727 |
| 1,024 | 0.950049 | 0.019175 |
| 65,536 | 0.950001 | 0.000305 |
| 1,000,000,000 | 0.950000 | 0.00000002 |

Gustafson's efficiency settles at 0.95. Amdahl's keeps falling towards zero.

### Edge cases

| Case | Gustafson | Amdahl |
|---|---|---|
| Serial share = 0 | `E = 1` for all P | `E = 1` for all P |
| Serial share = 1 | `E = 1/P → 0`, which equals `1 − α` | `E = 1/P → 0` |

The formulas agree in both edge cases. If nothing is serial, both are perfect. If everything is serial, both give no useful speedup.

## Part D: What the limits mean

Efficiency is the share of processor time spent on useful work.

**Gustafson:** the two terms of `E(P) = (1 − α) + α/P` can each be explained.

- `1 − α`: every processor is busy for this share of the run, in the parallel phase.
- `α/P`: during the serial phase, 1 of the P processors is busy.

As P grows, only the second term disappears. The first stays, because **a bigger problem keeps every processor busy in the parallel phase**.

**Amdahl:** with a fixed job, the parallel part gets shorter and shorter, but the serial part does not. Nearly all of the run becomes serial time, during which `P − 1` processors wait. Almost all of the machine's time is wasted, so efficiency goes to 0.

**The difference in one sentence:** in Gustafson's model, the share of time in the parallel phase stays constant, and in Amdahl's model, it shrinks to zero.

## Part E: Warnings

1. **`s` and `α` are measured in different ways** (Task 03). The table above uses the value 0.05 for each model to show the shapes of the curves. It is not two predictions for one run.
2. **`1 − α` holds only if `α` is constant.** In real systems `α` often grows with P because of communication. Then the real efficiency can fall below `1 − α` (Task 12).
3. **A good limit is not the same as a good result for every need.** A 95% efficient run is still only useful if the bigger problem is one you actually wanted to solve.

All numbers can be reproduced with `benchmarks/check_part1.py`.