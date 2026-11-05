# Task 15: Energy and Power Proportionality under Scaled Speedup

## Short answer

For the same wall-clock time, `P` cores use far more energy than 1 core, but they also do far more work. **Energy for each unit of work rises only a little under Gustafson's model**, and the extra is exactly the idle power wasted in the serial phase.

```text
Energy:                  E(P) = T · (P · P_idle + S(P) · P_active)
Energy per unit of work: e(P)/e(1) = (r/η(P) + 1) / (r + 1)
```

Here `η(P) = S(P)/P` is parallel efficiency and `r = P_idle/P_active`. I write `η` to avoid confusion with energy `E`.

## Assumptions

1. A node always draws static power `P_idle` while it is on.
2. A node doing computation draws an extra dynamic power `P_active`, so a busy node draws `P_idle + P_active`.
3. All nodes are on for the whole run, and the run lasts `T`.
4. All nodes run at a fixed clock speed.
5. Network, storage and cooling power are not counted.

**Be careful with definitions:** some sources use "active power" to mean the total power of a busy node. If so, replace `P_active` here with `(active − idle)`.

## Deriving the energy equation

The run has a serial phase of length `αT` and a parallel phase of length `(1 − α)T`.

| # | Statement | Why |
|---:|---|---|
| 1 | `E_static = P · P_idle · T` | All P nodes are on for time `T`. |
| 2 | Busy node-time in the serial phase `= 1 · αT` | One node works. |
| 3 | Busy node-time in the parallel phase `= P · (1 − α)T` | All P nodes work. |
| 4 | Total busy node-time `= T(α + (1 − α)P) = T · S(P)` | Add lines 2 and 3, then use Task 02. |
| 5 | `E_dynamic = P_active · T · S(P)` | Dynamic power is drawn only during busy time. |
| 6 | `E(P) = T(P · P_idle + S(P) · P_active)` | Add lines 1 and 5. |

**Key result:** static energy follows `P`, the number of nodes on. Dynamic energy follows `S(P)`, the work done.

## 1 core vs. P cores, with identical wall-clock time

For one core, `S(1) = 1`, so:

```text
E(1) = T(P_idle + P_active)
```

The ratio of total energy is:

```text
E(P)/E(1) = (P·r + S(P)) / (r + 1)
```

The two runs do different amounts of work (1 and `S(P)`), so we should also compare energy per unit of work:

```text
e(P) = E(P)/S(P) = T(P_idle/η(P) + P_active)
e(P)/e(1) = (r/η(P) + 1) / (r + 1)
```

The penalty comes only from `r/η(P)`. If efficiency is perfect (`η = 1`), the ratio is exactly 1.

## Worked example

With `α = 0.04`, `P_idle = 50 W`, `P_active = 100 W`, `T = 1 hour`, so `r = 0.5`.

```text
E(1) = 1 h × (50 + 100) W = 150 Wh = 0.15 kWh

P = 1,024:
S = 983.08
E = 1 h × (1024 × 50 + 983.08 × 100) W = 149,508 Wh = 149.51 kWh
```

| P | `S(P)` | `E(P)` (kWh) | Average power (kW) | Energy ratio | Work ratio | Energy per work ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.15 | 0.15 | 1.00 | 1.00 | 1.0000 |
| 16 | 15.40 | 2.34 | 2.34 | 15.60 | 15.40 | 1.0130 |
| 256 | 245.80 | 37.38 | 37.38 | 249.20 | 245.80 | 1.0138 |
| 1,024 | 983.08 | 149.51 | 149.51 | 996.72 | 983.08 | 1.0139 |
| 4,096 | 3,932.20 | 598.02 | 598.02 | 3,986.80 | 3,932.20 | 1.0139 |
| 16,384 | 15,728.68 | 2,392.07 | 2,392.07 | 15,947.12 | 15,728.68 | 1.0139 |

The total energy is about 997 times larger at `P = 1,024`, but the work is 983 times larger. **Each unit of work costs only 1.39% more.**

### Limit as P → ∞

Since `η(P) → 1 − α`:

```text
e(P)/e(1) → (r/(1 − α) + 1) / (r + 1)
```

With these values, the limit is 1.0139. So the penalty stays bounded for any P.

## Where the extra energy goes

Compare both ways of doing the same scaled job at `P = 1,024`:

| | Time | Energy |
|---|---:|---:|
| 1 core alone | 983.08 h | 147.46 kWh |
| 1,024 cores | 1 h | 149.51 kWh |

The difference is 2,046 Wh. In general:

```text
E(P) − S(P)·E(1) = T · P_idle · (P − S(P)) = T · P_idle · α(P − 1)
```

**Check:** `1 h × 50 W × 40.92 = 2,046 Wh` ✓

So the extra energy is exactly the static power of the processors that sit idle during the serial phase. For about 1.4% more energy, you get the answer 983 times sooner. This assumes the job can fit in one core's memory at all (Task 05 shows that it often cannot).

## Energy proportionality

A system is **energy proportional** if its energy use follows the work it does. Here, this depends on `r`:

| `r = P_idle/P_active` | Meaning | Energy per work ratio at `P = 1,024` |
|---:|---|---:|
| 0 | Perfectly proportional, no idle power | 1.000 |
| 0.5 | The example above | 1.014 |
| 2 | High idle power | 1.028 |

Lower idle power or a lower `α` both reduce the penalty.

**Unused nodes that stay on change the result.** The figures above assume the 1-core run has its other nodes switched off. Suppose instead that a whole 1,024-node machine stays on but only one node computes:

```text
E = 1 h × (1024 × 50 + 100) W = 51,300 Wh for 1 unit of work
```

That costs about 337 times more energy per unit of work than the full parallel run. **On a machine that is already on, idle nodes are the real waste, so using them is the efficient choice.**

## Comparison with Amdahl

The same per-work formula works for a fixed job, using Amdahl's efficiency. Here `s = 0.04` and `r = 0.5`.

| P | Gustafson `η` | Gustafson energy per work | Amdahl `η` | Amdahl energy per job |
|---:|---:|---:|---:|---:|
| 16 | 0.9625 | 1.013 | 0.6250 | 1.20 |
| 256 | 0.9602 | 1.014 | 0.0893 | 4.40 |
| 1,024 | 0.9600 | 1.014 | 0.0239 | 14.64 |
| 4,096 | 0.9600 | 1.014 | 0.0061 | 55.60 |
| 16,384 | 0.9600 | 1.014 | 0.0015 | 219.44 |

- **Under Gustafson**, the penalty is bounded, because efficiency stays near `1 − α`.
- **Under Amdahl**, the penalty grows without limit, because efficiency goes to 0 while all nodes stay on and draw static power.

So a fixed small job on too many nodes wastes energy, but a job that grows with the machine does not.

## Limits of this model

1. **Peak power can be the real limit.** The machine draws about 150 kW at `P = 1,024`, and a power budget can limit P even when energy per work is good.
2. **The clock speed is fixed here.** Real processors can reduce frequency and voltage, and dynamic power falls faster than speed. So many slower nodes can sometimes use less energy than fewer fast ones. This model does not capture that.
3. **Communication uses energy.** The network, and processors waiting on messages, add power that grows with P. Adding `C(P)` from Task 12 would make run time and static energy larger.
4. **`α` is assumed to be constant.** If it grows, efficiency falls and the energy penalty grows with it.

The numbers can be reproduced with `benchmarks/part3_calculations.py`. The power values are examples.
