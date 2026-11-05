# Task 10: Memory Growth Rate vs. Scaled Speedup

## Answer

The task does not say how many bytes each grid point uses, and the answer depends on it. So we state our assumptions:

- Each grid point stores **one 8-byte number** (`b = 8`).
- **1 GB = 2³⁰ bytes.**
- All 32 GB on each node holds grid data.
- The grid is split into equal cubic blocks, one per node.

| Nodes | Total memory | **Maximum `N`** | Grid points |
|---:|---:|---:|---:|
| 64 | 2,048 GB | **6,500** | 2.746 × 10¹¹ |
| 512 | 16,384 GB | **13,000** | 2.197 × 10¹² |

Having 8 times more nodes gives 8 times more grid points, but **only 2 times the side length**.

## Formulation

| Symbol | Meaning |
|---|---|
| `P` | Number of nodes |
| `c` | Memory per node, with `c ≤ 32 GB` |
| `b` | Bytes per grid point |
| `N` | Grid points along one side of the 3D grid |

The memory available grows with the nodes, and the memory needed grows with the grid:

```text
Available:  M(P) = c · P
Needed:     M(N) = b · N³
```

The grid must fit:

```text
b · N³ ≤ c · P
N ≤ (c · P / b)^(1/3)
```

**Growth rate:** since `c` and `b` are constants, `N_max ∝ P^(1/3)`. The grid point count `N³ ∝ P`, which matches `W(P) = P · W(1)`.

## Step 1: Limit from total memory

```text
P = 64:   total = 64 × 32 GB  = 2,048 GB  = 2⁴¹ bytes
          N ≤ (2⁴¹ / 8)^(1/3) = (2³⁸)^(1/3) = 6,501.99

P = 512:  total = 512 × 32 GB = 16,384 GB = 2⁴⁴ bytes
          N ≤ (2⁴⁴ / 8)^(1/3) = (2⁴¹)^(1/3) = 13,003.99
```

Rounding down to whole numbers gives 6,501 and 13,003. These are upper limits, because they treat all the memory as one large pool.

## Step 2: Limit from each node's strict cap

The task says the cap applies **per node**, so the largest block must fit on one node.

```text
Points per node ≤ 32 GB / 8 bytes = 2³² = 4,294,967,296
Block side n ≤ (2³²)^(1/3) = 1,625.50, so n = 1,625
```

Checking the whole-number limits:

| Block side | Memory used | Fits in 32 GB? |
|---:|---:|---|
| 1,625 | 31.971 GB | Yes |
| 1,626 | 32.030 GB | No |

Both node counts are perfect cubes, so the blocks fit together exactly:

```text
P = 64  = 4³:  N = 4 × 1,625 = 6,500
P = 512 = 8³:  N = 8 × 1,625 = 13,000
```

**Why not `N = 6,501`?** Split into 4 parts along a side, 6,501 points would put 1,626 points along that side in at least one block. A node holding a block of 1,626 × 1,626 × 1,626 would be over its cap.

Memory used at these sizes: 2,046.1 GB of 2,048 GB, and 16,368.9 GB of 16,384 GB.

## How the assumptions change the answer

Maximum `N` under a strict per-node cap:

| Bytes per point | What this could hold | 64 nodes (GB = 2³⁰) | 512 nodes (GB = 2³⁰) | 64 nodes (GB = 10⁹) | 512 nodes (GB = 10⁹) |
|---:|---|---:|---:|---:|---:|
| 8 | 1 number | 6,500 | 13,000 | 6,348 | 12,696 |
| 40 | 5 numbers | 3,800 | 7,600 | 3,712 | 7,424 |
| 80 | 10 numbers | 3,016 | 6,032 | 2,944 | 5,888 |

The ratio between 512 and 64 nodes is always 2, because `(512/64)^(1/3) = 2`. This part does not depend on the assumptions.

## What this means for scaled speedup

1. **Memory and work can grow at the same rate.** Work per node stays fixed and memory per node stays fixed, so weak scaling can continue as long as the block fits on a node.
2. **Resolution grows slowly.** Doubling the side length of a 3D grid needs 8 times the nodes.
3. **Memory is not the only limit.** In many simulations, a finer grid also needs more time steps. If the number of steps grows with `N`, going from 6,500 to 13,000 would **double the total run time**, even though each step takes about the same time. In that case the fixed-time promise fails, and the problem must grow more slowly than memory allows. Task 11 explores this.
4. **Real programs have less space.** The operating system, extra boundary layers and temporary arrays all need memory, so the real `N` is lower than these limits.

The numbers can be reproduced with `benchmarks/part2_calculations.py`.
