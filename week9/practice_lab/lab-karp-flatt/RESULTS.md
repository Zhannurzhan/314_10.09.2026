# Karp–Flatt Lab Results

**Student ID:** <230109009>
**Hardware:** <12th Gen Intel(R) Core(TM) i5-12450H>, <12> logical cores (from `os.cpu_count()`)
**OS:** Windows 11

## Measured Results

| Experiment | T1 (s) | T2 (s) | T4 (s) | S2 | S4 | e(2) | e(4) | Diagnosis |
|---|---|---|---|---|---|---|---|---|
| A: Serial Tail | 7.757 | 4.181 | 2.812 | 1.86x | 2.76x | 0.078 | 0.150 | e stays small → fixed Amdahl serial path (0.40 s sleep) |
| B: Lock Contention | 5.765 | 36.732 | 39.790 | 0.16x | 0.14x | 11.742 | 8.869 | e explodes, slowdown with more cores → lock thrashing & cache-line bouncing |
| C: Privatized State | 1.363 | 0.626 | 0.492 | 2.18x | 2.77x | -0.082 | 0.148 | e collapses, real speedup → scaling restored by privatization |

## Rubric Checks (computed from my numbers)

**Experiment A:** e(4) / e(2) = 0.150 / 0.078 = **1.92** (target range [0.80, 1.25]: **not met**)
- Both e values stay low (well under 0.5), so the loss comes from the sequential sleep and not from contention. e(4) is about twice e(2), which is likely caused by process-pool startup overhead growing with more workers on Windows.

**Experiment B:** T4 = 39.790 s > T1 = 5.765 s (**met**). e(4) >> e(2)? e(4) = 8.869 < e(2) = 11.742 (**not met**)
- Adding cores made the program about 7x slower and both e values are far above 1.50. This is the contention signature. The worst slowdown already appears at P=2, so e(2) ends up higher than e(4).

**Experiment C:** T4 = 0.492 s < T1 = 1.363 s (**met**). E4 = S4 / 4 = 2.77 / 4 = **69%** (target > 70%: **just below**)
- Removing the shared lock turned a 7x slowdown into a 2.77x speedup. The negative e(2) = -0.082 comes from slightly superlinear speedup at P=2 (2.18x), most likely measurement noise or warm-up effects in the P=1 run.

## Conclusion

- **A:** Low e(2) and e(4) → the limit is a fixed serial portion (Amdahl).
- **B:** Huge e and negative speedup → the limit is lock contention, not serial code.
- **C:** Privatizing the counter removes the contention and restores real parallel speedup.