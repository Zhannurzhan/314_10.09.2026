# Task 06: Parametric Speedup Across Processor Densities

**Given:** `α = 0.04`

**Find:** `S(P)` and `E(P)` for `P = 16, 64, 256, 1024, 4096, 16384`

## Formulas

```text
S(P) = P − α(P − 1)
E(P) = S(P) / P
```

## Results

| P | Lost to serial, `α(P − 1)` | `S(P)` | `E(P)` | `E(P)` in % |
|---:|---:|---:|---:|---:|
| 16 | 0.60 | 15.40 | 0.96250000 | 96.2500 |
| 64 | 2.52 | 61.48 | 0.96062500 | 96.0625 |
| 256 | 10.20 | 245.80 | 0.96015625 | 96.0156 |
| 1,024 | 40.92 | 983.08 | 0.96003906 | 96.0039 |
| 4,096 | 163.80 | 3,932.20 | 0.96000977 | 96.0010 |
| 16,384 | 655.32 | 15,728.68 | 0.96000244 | 96.0002 |

## Numeric log

```text
P = 16:     S = 16    − 0.04 × 15    = 16    − 0.60   = 15.40       E = 15.40 / 16        = 0.96250000
P = 64:     S = 64    − 0.04 × 63    = 64    − 2.52   = 61.48       E = 61.48 / 64        = 0.96062500
P = 256:    S = 256   − 0.04 × 255   = 256   − 10.20  = 245.80      E = 245.80 / 256      = 0.96015625
P = 1024:   S = 1024  − 0.04 × 1023  = 1024  − 40.92  = 983.08      E = 983.08 / 1024     = 0.96003906
P = 4096:   S = 4096  − 0.04 × 4095  = 4096  − 163.80 = 3932.20     E = 3932.20 / 4096    = 0.96000977
P = 16384:  S = 16384 − 0.04 × 16383 = 16384 − 655.32 = 15728.68    E = 15728.68 / 16384  = 0.96000244
```

## How efficiency degrades

From Task 04, efficiency can be written as:

```text
E(P) = (1 − α) + α/P = 0.96 + 0.04/P
```

| P | Gap above the limit, `E(P) − 0.96` |
|---:|---:|
| 16 | 0.00250000 |
| 64 | 0.00062500 |
| 256 | 0.00015625 |
| 1,024 | 0.00003906 |
| 4,096 | 0.00000977 |
| 16,384 | 0.00000244 |

1. **Efficiency hardly falls.** P grows 1,024 times from 16 to 16,384, but efficiency drops by only about 0.25 percentage points, from 96.25% to about 96.00%.
2. **The drop gets smaller each time.** Each step multiplies P by 4 and divides the gap by exactly 4.
3. **Efficiency never goes below 96%.** This is the limit `1 − α`.
4. **Speedup keeps growing in a nearly straight line.** Each extra processor adds about 0.96 to the speedup, because `S(P) = 0.04 + 0.96P`.

## What the table hides

- **The loss still grows in absolute terms.** At `P = 16,384`, about 655 processors' worth of time goes unused, even though the efficiency looks excellent.
- **`α` is assumed to be constant.** This is the model's best case. Real communication usually grows with P, so real efficiency would be lower at large P. Task 12 covers this.

The numbers can be reproduced with `benchmarks/part2_calculations.py`.
