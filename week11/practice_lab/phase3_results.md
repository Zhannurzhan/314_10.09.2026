### Part 3A: Roofline ceilings (p = P_phys = 8)

| Ceiling | Min ms | Median ms | IQR ms | Value |
|---|---:|---:|---:|---:|
| Memory bandwidth BW_sustained | 3.16 | 3.21 | 0.12 | 39.93 GB/s |
| Compute P_sustained | 1197.8 | 1218.0 | 26.8 | 0.3941 GFLOP/s |

Ridge point I* = P_sustained / BW_sustained = 0.3941 / 39.93 = 0.00987 FLOP/byte

### Part 3B: GEMM (N = 256, p = 8, 2N^3 = 33,554,432 FLOPs)

Correctness: max|C1 - C2| = 0.00e+00, max|C3 - C2| over all BS = 0.00e+00 (PASS, limit 1e-9)

| Variant | BS | Tile footprint 3*BS^2*8 | Fits in | Median ms | IQR ms | GFLOP/s | Intensity I (FLOP/B) | Roofline bound GFLOP/s | % of bound |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| V1 naive i-j-k | - | - | - | 403.5 | 25.1 | 0.0832 | 0.125 | 0.3941 | 21.1% |
| V2 reordered i-k-j | - | - | - | 472.9 | 13.6 | 0.0710 | 0.083 | 0.3941 | 18.0% |
| V3 blocked | 8 | 1.5 KB | L1d (all cores) | 595.2 | 49.7 | 0.0564 | 0.667 | 0.3941 | 14.3% |
| V3 blocked | 16 | 6.0 KB | L1d (all cores) | 534.6 | 45.4 | 0.0628 | 1.333 | 0.3941 | 15.9% |
| V3 blocked **(BS*)** | 32 | 24.0 KB | L1d (all cores) | 508.0 | 73.8 | 0.0661 | 2.667 | 0.3941 | 16.8% |
| V3 blocked | 64 | 96.0 KB | L2 | 514.0 | 31.4 | 0.0653 | 5.333 | 0.3941 | 16.6% |
| V3 blocked | 96 | 216.0 KB | L2 | 521.2 | 52.6 | 0.0644 | 8.000 | 0.3941 | 16.3% |
| V3 blocked | 128 | 384.0 KB | L2 | 510.8 | 21.6 | 0.0657 | 10.667 | 0.3941 | 16.7% |

Best tile BS* = 32: 0.0661 GFLOP/s (0.79x Variant 1, 0.93x Variant 2)
Every variant has I > I* = 0.00987, so all are on the COMPUTE roof.

VERIFY_HASH[PHASE3]: 7EA212C7714B0DFA (P_log=12)
