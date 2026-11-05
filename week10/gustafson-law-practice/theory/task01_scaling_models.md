# Task 01: Strong Scaling vs. Weak Scaling Foundations

## Short answer

Amdahl and Gustafson ask two different questions:

- **Amdahl (1967), strong scaling:** "I have one job of fixed size. How much faster can more processors finish it?"
- **Gustafson (1988), weak scaling:** "I have a fixed amount of time. How much more work can more processors finish in that time?"

Amdahl's answer has a ceiling. Gustafson's answer keeps growing with the number of processors, P.

## The two laws side by side

| | Amdahl (fixed-size) | Gustafson (fixed-time) |
|---|---|---|
| What stays constant | Total problem size | Run time on the parallel machine |
| What changes as P grows | Run time gets shorter | Problem size gets bigger |
| Work per processor | Gets smaller: `W/P → 0` | Parallel work per processor stays constant |
| Where the serial share is measured | On one processor (`s`) | On the parallel machine (`α`) |
| Speedup formula | `S = 1 / (s + (1 − s)/P)` | `S = P − α(P − 1)` |
| Speedup as `P → ∞` | Stops at `1/s` | Grows without limit, about `(1 − α)·P` |
| Also called | Strong scaling | Weak scaling, or scaled speedup |

## Assumptions of each model

### Amdahl (1967)

Amdahl's paper has no equations. He argued in words that part of every real program is hard to run in parallel, such as "data management housekeeping". He estimated this part at about 40% of executed instructions, which would limit throughput to about five to seven times the sequential rate. The formula we now call Amdahl's Law was written down later from this argument.

The formula assumes:

1. The problem size does not change when P changes.
2. A share `s` of the one-processor run time cannot be made parallel.
3. The rest, `1 − s`, divides perfectly among P processors.

### Gustafson (1988)

Gustafson said the assumption of fixed size is rarely true. He wrote that "in practice, the problem size scales with the number of processors" and that "it may be most realistic to assume that run time, not problem size, is constant." He credits this alternative to E. Barsis at Sandia.

The formula assumes:

1. The parallel machine's run time is fixed. We scale it so that `s + p = 1`.
2. Each processor does the same parallel work as before, so total parallel work grows with P.
3. The serial time `s` does not grow when the problem grows.

## What happens to the problem as P → ∞

Let `W` be total work and `N` be the size of the grid along one side (the problem dimension).

### Under Amdahl

```text
W(P) = W₀                     constant
N(P) = N₀                     constant
W(P)/P = W₀/P → 0             each processor gets almost nothing to do
T(P) = s + (1 − s)/P → s      time cannot go below the serial part
S(P) → 1/s                    the ceiling
```

### Under Gustafson

```text
W(P) = s + p·P → ∞            work grows in a straight line with P
parallel work per processor = p   constant
T(P) = s + p = 1              time never changes
S(P) = s + p·P → ∞            no ceiling
```

If the work per grid point is constant, a grid in `d` dimensions has `W ∝ N^d`. So:

```text
N(P) ≈ N₁ · P^(1/d)
```

A 3D grid doubles its side length only when P grows 8 times. So the total work grows like P, but the side length `N` grows more slowly.

## Why the names "fixed-size" and "fixed-time"

**Amdahl is fixed-size** because the size of the job is the constant. Speedup is measured as time saved on that same job.

**Gustafson is fixed-time** because the wall-clock time is the constant. Speedup is measured as extra work completed in that same time.

## A worked example

Here the serial share is 5% in both models, measured in each model's own way (`s = α = 0.05`).

| P | Amdahl time | Amdahl speedup | Gustafson time | Gustafson work | Gustafson speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.00 | 1 | 1.00 | 1.00 |
| 16 | 0.1094 | 9.14 | 1 | 15.25 | 15.25 |
| 256 | 0.0537 | 18.62 | 1 | 243.25 | 243.25 |
| 1,024 | 0.0509 | 19.64 | 1 | 972.85 | 972.85 |
| 65,536 | 0.0500 | 19.99 | 1 | 62,259.25 | 62,259.25 |

Amdahl's speedup stops near `1/0.05 = 20`. Gustafson's keeps growing. These numbers come from `benchmarks/check_part1.py`.

## The divergence between the two laws

### Physical

- **Amdahl** fits work with a real fixed size, like one image to process or one query to answer fast. Extra processors have less and less to do, and eventually most of them wait for the serial part.
- **Gustafson** fits work that can always use more detail, like a weather model with a finer grid. Each new node also brings memory, so a bigger problem can fit. Processors stay busy.

### Philosophical

- **Amdahl's view:** the problem is fixed and the computer serves it. Parallelism has a built-in limit, so faster single processors matter most.
- **Gustafson's view:** the time a person will wait is fixed, and the problem expands to fill it. Parallelism is worth building even at very large sizes.

## They do not contradict each other

Both are correct under their own assumptions. Task 03 shows that one formula can be converted exactly into the other. The right question is which one matches your workload:

- A job with a fixed size that you want done faster: use Amdahl.
- A job you would make larger if you had more machine: use Gustafson, then check that the serial time really does stay constant.

## References

1. G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," *AFIPS Conference Proceedings*, vol. 30, pp. 483–485, 1967.
2. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988.