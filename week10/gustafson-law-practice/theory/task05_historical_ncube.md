# Task 05: Deconstructing the "Amdahl's Pessimism" Fallacy

## Summary

In 1988, Gustafson, Montry and Benner at Sandia National Laboratories reported scaled speedups above 1,000 on a 1,024-processor NCUBE hypercube. If you put their 0.4% serial fraction into Amdahl's formula, you get a speedup of only about 201, so the result looks impossible.

It was not impossible, because the two numbers belong to different frames of reference. The 0.4% was measured on the parallel machine for a problem that grew with P. Measured on one processor, the same serial share is only a few parts per million. Amdahl's mathematics was not wrong. The real error was assuming that problem size stays fixed when machines get bigger.

## Historical background

- **The challenge:** in November 1985, Alan Karp challenged the community to show "a speedup of at least 200 for a real scientific application on a general-purpose, MIMD computer."
- **The expectation:** the judges of the first Gordon Bell Prize had expected a winning speedup of 50 or less.
- **The result:** the Sandia team won the Bell Award's first place and Karp's prize. They published the work in *SIAM J. Sci. Stat. Comput.* in July 1988. Gustafson's short note "Reevaluating Amdahl's Law" appeared in *Communications of the ACM* in May 1988.

## The machine

The system was a 1,024-node NCUBE/ten hypercube (the task sheet calls it nCUBE 10).

| Property | Value |
|---|---|
| Memory per node | 512 KB |
| Floating-point operation | 7 to 15 µs |
| Single-node Fortran speed | 0.07 to 0.13 MFLOPS |
| Message cost | About 0.35 ms to start, then about 2 µs per byte |

## The experiment

There were three real applications, and each was run in two ways.

**1. Fixed-size runs.** The problem stayed the same and the processor count changed. This tests Amdahl's question.

**2. Scaled runs.** The work per processor stayed the same, so the problem grew with P. This tests Gustafson's question. On 1,024 processors:

- **Wave mechanics:** 192 × 192 points per node, giving a 6,144 × 6,144 global grid.
- **Fluid dynamics:** a 2,048 × 2,048 global grid.
- **Beam strain analysis:** a 64 × 32 subdomain per node, giving about two million finite elements.

### Results from the SIAM paper (1,024 processors)

| Application | Fixed-size speedup | Fixed-size `s` | Scaled speedup | Scaled `s′` | Equivalent one-processor `s` |
|---|---:|---:|---:|---:|---:|
| Wave mechanics | 637 | 0.0006 | 1,020 | 0.0034 | 0.000003 |
| Fluid dynamics | 519 | 0.001 | 1,009 | 0.013 | 0.00001 |
| Beam strain | 502 | 0.001 | 1,019 | 0.005 | 0.000005 |

The authors' scaled efficiency was 0.987 to 0.996.

## Why it looked impossible

Using Amdahl's formula on 1,024 processors:

| Serial fraction `s` | Speedup | Ceiling `1/s` |
|---:|---:|---:|
| 1% | 91.2 | 100 |
| 0.8% | 111.5 | 125 |
| 0.4% | 201.1 | 250 |
| 0.00235% | 1,000 | about 42,600 |

To reach 1,000 under Amdahl with P = 1,024, the serial fraction must be about **0.00235%**. That is 170 times smaller than 0.4%. Gustafson described the steepness of Amdahl's curve near `s = 0` by saying it implies "very few problems will experience even a 100-fold speedup."

## How problem expansion removed the ceiling

Gustafson's formula with the serial fraction measured on the parallel machine:

```text
α = 0.4%:  S = 1024 − 0.004 × 1023 = 1019.9
α = 0.8%:  S = 1024 − 0.008 × 1023 = 1015.8
```

These match the speedups of about 1,020 and 1,016 in Gustafson's note. The same 0.4%, seen on one processor (Task 03), is:

```text
f = 0.004 / (0.004 + 0.996 × 1024) = 0.00039%
```

That is far below the 0.00235% that Amdahl's formula needs for a speedup of 1,000. So there was never a contradiction.

The wave application shows this clearly. Its overhead was about 44 seconds, out of a calculated one-processor time of 13.1 million seconds. The overhead is small only because the problem is so large.

## Critical review: what the experiment did and did not prove

**1. Amdahl's formula was not disproved.** The fixed-size results fit it well:

| Application | Fixed `s` | Amdahl's formula gives | Measured |
|---|---:|---:|---:|
| Wave mechanics | 0.0006 | 635 | 637 |
| Fluid dynamics | 0.001 | 506 | 519 |
| Beam strain | 0.001 | 506 | 502 |

The authors obtained these `s` values from the measured speedups, so this close fit is partly expected. The small differences come from the rounding of `s`. Even so, fixed-size speedup stayed near 500 to 640 and scaled speedup passed 1,000 on the same machine. The word "dismantled" is too strong: the experiment changed the question being asked.

**2. The "1,000×" uses a one-processor time that was never measured.** One node had only 512 KB, so it could not hold the scaled problems. The authors extrapolated the one-processor times: about 5 months for wave, 3.6 years for fluid, and 20 years for beam. They measured the communication and idle time on the real machine, then based speedup on those measurements. The method is reasonable, but it does not compare two real runs. The Bell Prize report calls the fixed-size figures lower bounds and notes that the scaled figures rely on a model of idle time, validated to within a few percent.

**3. The applications suited this kind of scaling.** The wave and fluid applications used nearest-neighbour communication. The beam code did need global inner products, but their cost grew only with `log₂P`. Other kinds of algorithm scale much less well (Tasks 13 and 20).

**4. Scaling needed a problem worth enlarging.** The result is useful because more grid points gave more physical detail. It says nothing for a job whose size is truly fixed.

## Note on differing numbers

Sources do not give identical figures, so always say which one you are quoting.

| Source | Scaled speedups | Serial fraction | Fixed-size speedups |
|---|---|---|---|
| CACM note (May 1988) | 1,021 / 1,020 / 1,016 | 0.4–0.8% | Not listed |
| SIAM paper (July 1988) | 1,019 / 1,020 / 1,009 | 0.34–1.3% | 502 / 637 / 519 |
| Bell Prize report | About 1,021 / 1,020 / 1,016 | Not listed | 452 / 637 / 519 |

Each row lists the applications in the order beam, wave, fluid.

The task sheet's "0.4%" is a value from the CACM note's range. It is not a figure for all three applications.

## Conclusion

"Amdahl's pessimism" came from one assumption: that the problem size stays fixed. The Sandia experiment showed that this assumption does not hold for much of scientific computing. It left Amdahl's mathematics intact: for a fixed job, the ceiling is still real. The lasting lesson is to ask first whether a problem will grow with the machine, then choose the model that fits.

The formula values can be reproduced with `benchmarks/check_part1.py`.

## References

1. J. L. Gustafson, G. R. Montry, R. E. Benner, "Development of Parallel Methods for a 1024-Processor Hypercube," *SIAM J. Sci. Stat. Comput.*, vol. 9, no. 4, pp. 609–638, July 1988. <https://corpora.tika.apache.org/base/docs/govdocs1/803/803231.pdf>
2. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, vol. 31, no. 5, May 1988. <https://users.cs.utah.edu/~hari/teaching/paralg/Gustafson88.pdf>
3. G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," *AFIPS Conference Proceedings*, vol. 30, pp. 483–485, 1967. <https://www3.cs.stonybrook.edu/~rezaul/Spring-2019/CSE613/reading/Amdahl-1967.pdf>
4. Gordon Bell Prize report on the 1987 awards. <https://netlib.org/benchmark/bell1>