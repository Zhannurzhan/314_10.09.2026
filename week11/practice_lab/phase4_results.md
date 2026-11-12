### Phase 4: Skewed workload (N = 8,000, p = 8)

Total work = 106,651,060 heavy iterations. The last contiguous block (items 7000-7999) holds 35,204,198 = 33.0% of it (a fair share is 12.5%).

| Strategy | Min ms | Median T_wall ms | IQR ms | Speedup vs block | Imbalance lambda |
|---|---:|---:|---:|---:|---:|
| Block static | 6435.6 | 6993.4 | 1231.2 | 1.00x | 2.689 |
| Cyclic static | 3472.1 | 3551.7 | 73.7 | 1.97x | 1.029 |
| Dynamic C=1 **(C*)** | 3441.6 | 3484.8 | 71.0 | 2.01x | 1.001 |
| Dynamic C=8 | 3464.8 | 3589.7 | 124.3 | 1.95x | 1.009 |
| Dynamic C=64 | 14717.8 | 14766.7 | 74.6 | 0.47x | 1.043 |
| Dynamic C=512 | 4692.8 | 4935.5 | 450.0 | 1.42x | 1.522 |

**Per-worker active time t_worker[t] in ms (median run)**

| Worker | Block static | Cyclic static | Dynamic C=1 |
|---:|---:|---:|---:|
| 0 | 68.8 | 3290.3 | 3483.5 |
| 1 | 357.0 | 3416.9 | 3484.8 |
| 2 | 762.7 | 3450.6 | 3483.5 |
| 3 | 1623.1 | 3426.9 | 3480.3 |
| 4 | 2332.1 | 3551.0 | 3479.6 |
| 5 | 3643.3 | 3483.6 | 3479.5 |
| 6 | 5025.6 | 3479.0 | 3480.4 |
| 7 | 6992.9 | 3514.4 | 3478.6 |

Best chunk size C* = 1: T_wall = 3484.8 ms, 2.01x faster than block static.

VERIFY_HASH[PHASE4]: C14571F485034878 (P_log=12)
