# Task 03: The Serial Fraction Invariance Paradox

**Memo topic:** why the place where we measure the serial fraction changes the conclusion about scaling.

## The question

Amdahl says a small serial share puts a hard ceiling on speedup. Gustafson says a serial share of the same size barely hurts. How can both be right?

## Short answer

They use the same words for two different numbers:

- **`f` (Amdahl's `s`)** is the serial share of the time on **one processor**.
- **`α` (Gustafson's `s`)** is the serial share of the time on the **parallel machine**.

A serial share of 4% on 1,024 processors is only about 0.004% of the same job on one processor. The serial code did not change. What changed is the total time we compare it with.

## The same run, seen from two places

Take a scaled run with `α = 0.04` on `P = 1024`.

```text
Parallel machine (total time = 1):
  serial 0.04 | parallel 0.96

One processor doing the same work (total time = 983.08):
  serial 0.04 | parallel 0.96 × 1024 = 983.04
```

The serial time is 0.04 in both. But:

```text
On the parallel machine:  α = 0.04 / 1      = 4%
On one processor:         f = 0.04 / 983.08 = 0.0041%
```

## Converting between the two frames

### From α to f (scaled job)

The serial time is `α`, and one processor needs `α + (1 − α)P` in total (Task 02). So:

```text
f(P) = α / (α + (1 − α)P)
```

### From f to α (fixed job)

A fixed job on P processors takes `f + (1 − f)/P`, of which `f` is serial. So:

```text
α(P) = f / (f + (1 − f)/P) = fP / (fP + 1 − f)
```

## Proof: Amdahl with the right `f` gives Gustafson

Put `f(P)` into Amdahl's formula. Call `D = α + (1 − α)P` to make it easier to read, so `f = α/D` and `1 − f = (1 − α)P/D`.

| # | Statement | Why |
|---:|---|---|
| 1 | `S = 1 / (f + (1 − f)/P)` | Amdahl's formula |
| 2 | `f + (1 − f)/P = α/D + (1 − α)/D` | Substitute `f` and `1 − f`. The P cancels in the second term. |
| 3 | `= (α + 1 − α)/D = 1/D` | Add the fractions. |
| 4 | `S = D = α + (1 − α)P` | Line 1 with line 3 |
| 5 | `S = P − α(P − 1)` | This is Gustafson's formula (Task 02). ∎ |

**So the laws agree mathematically.** They only look different because each holds a different quantity constant.

## The paradox: both fractions cannot be constant

| Case | Held constant | The other fraction | Result |
|---|---|---|---|
| Scaled job (Gustafson) | `α` | `f(P) → 0` | The Amdahl ceiling rises forever |
| Fixed job (Amdahl) | `f` | `α(P) → 1` | Processors spend nearly all their time waiting |

Numbers for the value 0.04 in each case:

| P | Scaled job (α = 0.04): f(P) | Speedup | Fixed job (f = 0.04): α(P) | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.04 | 1.00 | 0.0400 | 1.00 |
| 16 | 0.002597 | 15.40 | 0.4000 | 10.00 |
| 256 | 0.0001627 | 245.80 | 0.9143 | 22.86 |
| 1,024 | 0.00004069 | 983.08 | 0.9771 | 24.43 |
| 4,096 | 0.00001017 | 3,932.20 | 0.9942 | 24.85 |
| 65,536 | 0.0000006358 | 62,914.60 | 0.9996 | 24.99 |

## How this removes the contradiction

For any job, Amdahl's ceiling is `1/f`. For a scaled job, this gives:

```text
ceiling(P) = 1/f(P) = (α + (1 − α)P) / α = S(P) / α
```

So `S(P) = α × ceiling(P)`. The speedup is always below the ceiling, as Amdahl requires. But the ceiling itself rises with P. **Amdahl's limit is never broken. It just moves away as the problem grows.**

## Critical points

1. **This does not make serial code harmless.** It only works if the serial time stays about the same while the parallel work grows. Consider the general case:

   ```text
   f(N) = serial work(N) / (serial work(N) + parallel work(N))
   ```

   If serial work grows as fast as parallel work, `f` does not fall, and making the problem bigger does not help.

2. **Real `α` is not constant.** Communication and synchronization usually grow with P, so `α` must be measured at each P. Task 12 extends the model for this.

3. **Do not mix the frames.** Putting `α` into Amdahl's formula answers no real question. This is the source of the apparent impossibility in Task 05.

4. **`f` for a scaled job is often not measured directly.** It usually depends on a one-processor time that was calculated, because one processor cannot hold the full problem.

## Conclusion

The serial overhead does not "stop mattering". Its time stays the same while the total work around it grows. Measured on one processor, its share shrinks towards zero, so it stops limiting the speedup. So the disagreement is about the frame of reference, not about the mathematics.

All numbers can be reproduced with `benchmarks/check_part1.py`.

## References

1. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.
2. J. L. Gustafson, G. R. Montry, R. E. Benner, "Development of Parallel Methods for a 1024-Processor Hypercube," *SIAM J. Sci. Stat. Comput.*, vol. 9, no. 4, pp. 609–638, July 1988.