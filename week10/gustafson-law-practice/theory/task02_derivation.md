# Task 02: Formal Derivation of Scaled Speedup

**Goal:** starting from a run on the parallel machine, prove that

```text
S(P) = s + p·P = P − α(P − 1)
```

## Assumptions and symbols

| Symbol | Meaning |
|---|---|
| `P` | Number of processors |
| `s` | Time of the serial phase in the parallel run (one processor works) |
| `p` | Time of the parallel phase in the parallel run (all P processors work) |
| `α` | Serial fraction: the share of the parallel run time that is serial |
| `T_P` | Time on P processors |
| `T_1` | Time for one processor to do the same total work |

We also assume:

1. All processors have the same speed.
2. In the parallel phase, every processor is busy for the full time `p`.
3. Communication and waiting are already counted inside `s`, so there is no other overhead.

## Derivation, line by line

| # | Statement | Why this is true |
|---:|---|---|
| 1 | `T_P = s + p` | The run has a serial phase, then a parallel phase. |
| 2 | `s + p = 1` | We choose the run time as our unit of time. This is the normalization given in the task. |
| 3 | `W_serial = 1 · s = s` | Work = processors × time. One processor works for time `s`. |
| 4 | `W_parallel = P · p` | P processors each work for time `p`. |
| 5 | `W_total = s + P·p` | Total work is the sum of lines 3 and 4. |
| 6 | `T_1 = s + P·p` | One processor at the same speed must do all of the work one piece at a time. This is the **equivalent serial workload**. |
| 7 | `S(P) = T_1 / T_P` | This is the definition of speedup. |
| 8 | `S(P) = (s + P·p) / (s + p)` | Substitute lines 1 and 6 into line 7. |
| 9 | `S(P) = s + p·P` | Line 2 makes the bottom equal 1. **This is the first form.** |
| 10 | `α = s / (s + p) = s` | Definition of the serial fraction, then line 2. |
| 11 | `p = 1 − α` | From lines 2 and 10. |
| 12 | `S(P) = α + (1 − α)·P` | Substitute lines 10 and 11 into line 9. |
| 13 | `S(P) = α + P − α·P` | Multiply out the bracket. |
| 14 | `S(P) = P − α·P + α` | Reorder the terms. |
| 15 | `S(P) = P − α(P − 1)` | Take out the common factor `α`. **This is the second form.** ∎ |

## A second proof, by counting idle time

This proof gives the meaning of the formula.

1. In a run of length 1, the machine offers `P × 1 = P` units of processor time.
2. In the serial phase of length `α`, one processor works and the other `P − 1` are idle.
3. So the idle time is `α(P − 1)`.
4. All other time is useful work, so useful work `= P − α(P − 1)`.
5. One processor would need exactly that much time to do this work, so `S(P) = P − α(P − 1)`. ∎

**Meaning:** the speedup you lose compared with the perfect value `P` is exactly the processor time wasted in the serial phase.

## Checking the result

| Test | Expected | Formula gives |
|---|---|---|
| `P = 1` | Speedup of 1 | `1 − α(0) = 1` ✓ |
| `α = 0` (no serial part) | Perfect speedup | `P − 0 = P` ✓ |
| `α = 1` (all serial) | No speedup | `P − (P − 1) = 1` ✓ |
| `α = 0.1, P = 10` | Both forms agree | `0.1 + 0.9·10 = 9.1` and `10 − 0.1·9 = 9.1` ✓ |

## Efficiency follows directly

Divide by P:

```text
E(P) = S(P)/P = 1 − α·(P − 1)/P
```

This matches the efficiency formula on the task sheet.

## Important notes

- **`T_1` is usually a calculated value.** For a big scaled problem, one processor often does not have enough memory to run it. Line 6 tells us what the time would be if it could. Task 05 shows that this happened in the original experiments.
- **The formula needs `s` to stay constant as P grows.** If the serial or communication time grows with P, `α` is not a constant and the formula becomes too optimistic. Task 12 handles this.
- **`s` here is not the `s` in Amdahl's Law.** Here it is measured on the parallel machine. Task 03 explains why this matters.

## Reference

J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.