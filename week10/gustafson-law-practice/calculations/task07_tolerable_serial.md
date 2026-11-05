# Task 07: Reverse Engineering Maximum Tolerable Serial Overhead

**Given:** `P = 8,192` and `S(P) ≥ 7,500`

**Find:** the largest allowed serial fraction, `α_max`

## Answer

```text
α_max = 692 / 8191 ≈ 0.084483 (about 8.45%)
```

## Derivation

| # | Statement | Why |
|---:|---|---|
| 1 | `P − α(P − 1) ≥ S_target` | Gustafson's formula must meet the requirement. |
| 2 | `P − S_target ≥ α(P − 1)` | Add `α(P − 1)` to both sides and subtract `S_target` from both sides. |
| 3 | `α ≤ (P − S_target) / (P − 1)` | Divide both sides by `P − 1`, which is positive, then read the line from right to left. |
| 4 | `α_max = (P − S_target) / (P − 1)` | The largest value that satisfies line 3. This is the general formula. |

Now substitute the numbers:

```text
α_max = (8192 − 7500) / (8192 − 1)
      = 692 / 8191
      = 0.08448297...
```

The fraction `692/8191` cannot be simplified, because 8191 is a prime number and does not divide 692.

## Checking the answer

```text
S = 8192 − (692/8191) × 8191 = 8192 − 692 = 7500  ✓
```

Why this is a maximum and not a minimum: when `α` gets bigger, `S` gets smaller, because each increase in `α` removes `(P − 1)` times that amount from `S`. So any `α` above `α_max` fails.

## Be careful when rounding

For a maximum limit, always round **down**. Rounding up breaks the requirement.

| `α` | `S(8192)` | Meets `S ≥ 7500`? |
|---:|---:|---|
| 0.08448 (rounded down) | 7,500.024 | Yes |
| 0.08449 (rounded up) | 7,499.942 | No |
| 0.0845 (rounded up) | 7,499.860 | No |

A safe practical limit is **`α ≤ 0.08448`**.

## What this means

- **Required efficiency:** `E = 7500 / 8192 = 0.915527`, or about 91.55%.
- **In real time:** in a 1-hour run, the serial part can take at most `0.08448297 × 3600 ≈ 304.1` seconds, about 5 minutes.
- **Allowed loss:** the machine can lose at most the equivalent of `8192 − 7500 = 692` processors.

## Comparison with Amdahl

For the same speedup under Amdahl's formula, a fixed job needs its one-processor serial fraction to be:

```text
s = α_max / 7500 = 692 / (7500 × 8191) ≈ 0.0000113 (about 0.00113%)
```

Both limits describe the same run, seen in two frames (Task 03). The 8.45% is measured on the parallel machine, and the 0.00113% is measured on one processor. If the problem cannot grow, the Amdahl limit is the one that applies, and it is 7,500 times tighter.

## Assumption

This assumes `α` is measured at `P = 8,192` itself. A value measured on a smaller run may be too low, because communication usually takes a larger share as P grows.

The numbers can be reproduced with `benchmarks/part2_calculations.py`.
