#!/usr/bin/env python3
"""Check every number used in the Part I theory files (tasks 01-05).

No extra packages are needed. Run it with:
    python3 benchmarks/check_part1.py      (Mac / Linux)
    python  benchmarks\\check_part1.py      (Windows)
"""


def amdahl(s, p):
    """Fixed-size speedup. s = serial fraction on ONE processor."""
    return 1 / (s + (1 - s) / p)


def gustafson(a, p):
    """Scaled speedup. a = serial fraction on the PARALLEL machine."""
    return p - a * (p - 1)


def f_from_alpha(a, p):
    """Serial share of the scaled job if ONE processor ran all of it."""
    return a / (a + (1 - a) * p)


def alpha_from_f(f, p):
    """Serial share of parallel run time for a FIXED job on p processors."""
    return f * p / (f * p + 1 - f)


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol * max(1, abs(x), abs(y))


# --- Task 01: what happens to time and work as P grows (serial share 5%) ---
print("TASK 01: s = alpha = 0.05")
print(f"{'P':>8} {'Amdahl time':>12} {'Amdahl S':>10} {'Gust. time':>11} "
      f"{'Gust. work':>11} {'Gust. S':>10}")
for p in [1, 16, 256, 1024, 65536]:
    a = 0.05
    print(f"{p:>8} {a + (1 - a) / p:>12.4f} {amdahl(a, p):>10.2f} "
          f"{1.0:>11.4f} {a + (1 - a) * p:>11.2f} {gustafson(a, p):>10.2f}")
print("Amdahl ceiling 1/s =", round(1 / 0.05, 2))

# --- Task 02: the two forms of the formula are the same ---
print("\nTASK 02: s + p*P equals P - alpha*(P-1)")
for a in [0.0, 0.04, 0.1, 0.5, 1.0]:
    for p in [1, 2, 10, 1024]:
        assert close(a + (1 - a) * p, gustafson(a, p))
print("alpha=0.1, P=10: s + p*P =", round(0.1 + 0.9 * 10, 4),
      "| P - alpha*(P-1) =", round(gustafson(0.1, 10), 4))

# --- Task 03: moving between the two frames of reference ---
print("\nTASK 03: alpha is fixed at 0.04 on the parallel machine")
print(f"{'P':>8} {'f(P)':>12} {'Amdahl using f(P)':>18} {'Gustafson':>10}")
for p in [1, 16, 256, 1024, 4096, 65536]:
    f = f_from_alpha(0.04, p)
    assert close(amdahl(f, p), gustafson(0.04, p))
    assert close(alpha_from_f(f, p), 0.04)
    print(f"{p:>8} {f:>12.3e} {amdahl(f, p):>18.2f} {gustafson(0.04, p):>10.2f}")

print("\nTASK 03: f is fixed at 0.04 on one processor (a fixed job)")
print(f"{'P':>8} {'alpha(P)':>10} {'Amdahl S':>10}")
for p in [1, 16, 256, 1024, 4096, 65536]:
    print(f"{p:>8} {alpha_from_f(0.04, p):>10.4f} {amdahl(0.04, p):>10.2f}")

# --- Task 04: efficiency limits (serial share 5%) ---
print("\nTASK 04: efficiency with s = alpha = 0.05")
print(f"{'P':>8} {'E Gustafson':>12} {'E Amdahl':>10}")
for p in [1, 16, 256, 1024, 65536, 10**9]:
    eg = gustafson(0.05, p) / p
    ea = amdahl(0.05, p) / p
    assert close(eg, 0.95 + 0.05 / p)
    assert close(ea, 1 / (0.05 * p + 0.95))
    print(f"{p:>8} {eg:>12.6f} {ea:>10.8f}")
half = 1 / 0.05 + 1
assert close(amdahl(0.05, half) / half, 0.5)
print("Amdahl efficiency reaches 50% at P =", round(half))

# --- Task 05: the 1024-processor NCUBE numbers ---
print("\nTASK 05: what Amdahl's formula gives on 1024 processors")
for s in [0.01, 0.008, 0.004]:
    print(f"s = {s:<6} S = {amdahl(s, 1024):7.1f}   ceiling 1/s = {1 / s:6.1f}")
need = (1 / 1000 - 1 / 1024) / (1 - 1 / 1024)
assert close(amdahl(need, 1024), 1000)
print(f"s needed for S = 1000 on 1024 processors: {need:.3e} ({need * 100:.5f} %)")
print("Gustafson with alpha = 0.4% :", round(gustafson(0.004, 1024), 1))
print("Gustafson with alpha = 0.8% :", round(gustafson(0.008, 1024), 1))
f = f_from_alpha(0.004, 1024)
print(f"alpha = 0.4% seen on one processor: f = {f * 100:.5f} %")
print(f"s needed is {0.004 / need:.0f} times smaller than 0.4%; "
      f"its ceiling 1/s = {1 / need:.0f}")
print(f"Wave overhead share: 44 s / 13.1 million s = {44 / 13.1e6:.2e}")

print("\nReported values from the 1988 SIAM paper:")
print(f"{'problem':>7} {'s (fixed)':>10} {'Amdahl':>7} {'paper':>6} "
      f"{'s-prime':>8} {'Gustafson':>10} {'paper':>6} {'f(1024)':>10}")
rows = [("wave", 0.0006, 637, 0.0034, 1020),
        ("fluid", 0.001, 519, 0.013, 1009),
        ("beam", 0.001, 502, 0.005, 1019)]
for name, s, fixed, sp, scaled in rows:
    print(f"{name:>7} {s:>10} {amdahl(s, 1024):>7.0f} {fixed:>6} "
          f"{sp:>8} {gustafson(sp, 1024):>10.1f} {scaled:>6} "
          f"{f_from_alpha(sp, 1024):>10.2e}")

print("\nAll checks passed.")
