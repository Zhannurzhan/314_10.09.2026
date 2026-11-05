# Task 08: Cross-Over Point Determination (Amdahl vs. Gustafson)

**Given:** `s = α = 0.08` and `P = 128`

## Results

| Quantity | Value |
|---|---:|
| Amdahl speedup | **11.47** |
| Gustafson speedup | **117.84** |
| Ratio, Gustafson / Amdahl | **10.27 times** |
| Gustafson is higher than Amdahl by | **927.42%** |
| Amdahl is lower than Gustafson by | **90.27%** |

## Step 1: Amdahl

```text
S = 1 / (s + (1 − s)/P)
  = 1 / (0.08 + 0.92/128)
  = 1 / (0.08 + 0.0071875)
  = 1 / 0.0871875
  = 11.4695
```

## Step 2: Gustafson

```text
S = P − α(P − 1)
  = 128 − 0.08 × 127
  = 128 − 10.16
  = 117.84
```

## Step 3: Discrepancy

The percentage depends on which number you compare against, so both are given.

```text
Compared with Amdahl:     (117.84 − 11.4695) / 11.4695 × 100 = 927.42%
Compared with Gustafson:  (117.84 − 11.4695) / 117.84  × 100 =  90.27%
Ratio:                     117.84 / 11.4695                  =  10.2742
```

The task asks for "variance ratios". Here this means the ratio between the two predictions. It is not statistical variance.

## How the gap grows with P

| P | Amdahl | Gustafson | Ratio G/A | % above Amdahl |
|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.00 | 1.0000 | 0.00 |
| 2 | 1.8519 | 1.92 | 1.0368 | 3.68 |
| 8 | 5.1282 | 7.44 | 1.4508 | 45.08 |
| 32 | 9.1954 | 29.52 | 3.2103 | 221.03 |
| 128 | 11.4695 | 117.84 | 10.2742 | 927.42 |
| 512 | 12.2254 | 471.12 | 38.5361 | 3,753.61 |
| 2,048 | 12.4302 | 1,884.24 | 151.5856 | 15,058.56 |
| 16,384 | 12.4912 | 15,073.36 | 1,206.7152 | 120,571.52 |

## Is there a cross-over point?

The title of the task says "cross-over", but **the curves never cross**. They only touch at `P = 1`.

Let `q = 1 − s`, and write each speedup in terms of `s`, `q` and `P`:

| # | Statement | Why |
|---:|---|---|
| 1 | `S_G = s + qP` | Gustafson |
| 2 | `S_A = P / (sP + q)` | Amdahl, top and bottom multiplied by P |
| 3 | `S_G / S_A = (s + qP)(sP + q) / P` | Divide line 1 by line 2. |
| 4 | `S_G / S_A = s² + q² + sq(P + 1/P)` | Multiply out and divide by P. |
| 5 | `P + 1/P ≥ 2`, and it equals 2 only at `P = 1` | Standard result, since `P + 1/P − 2 = (P − 1)²/P ≥ 0` |
| 6 | At `P = 1`: `S_G / S_A = s² + q² + 2sq = (s + q)² = 1` | Line 4 with `P + 1/P = 2`, and `s + q = 1` |
| 7 | For `P > 1` and `0 < s < 1`: `S_G / S_A > 1` | From lines 4 to 6 |

So for `0 < s < 1`, Gustafson's value is always higher when `P > 1`. For large P, the ratio is about `s(1 − s)P`, so it grows in a straight line with P.

**Check at `P = 128`:**

```text
0.0064 + 0.8464 + 0.0736 × (128 + 1/128) = 10.2742  ✓
```

## Why they diverge

1. **The limits are different.** Amdahl can never exceed `1/s = 12.5`. At `P = 128`, it has already reached 91.76% of this limit. Gustafson keeps growing by `1 − α = 0.92` per processor.
2. **They do not describe the same run.** The value 0.08 means something different in each formula:
   - A **fixed job** with `f = 0.08` on one processor spends `α(128) = 91.76%` of its time on 128 processors in the serial part.
   - A **scaled job** with `α = 0.08` on 128 processors has a one-processor serial share of only `f(128) = 0.0679%`.
3. **The processors are used differently.** On the fixed job, processors finish quickly and wait for the serial part. On the scaled job, new work keeps them busy.

**Main point:** the 927% is not an error in either law. It measures how much the two assumptions differ. To compare them fairly on one real run, convert the serial fraction first (Task 03).

The numbers can be reproduced with `benchmarks/part2_calculations.py`.
