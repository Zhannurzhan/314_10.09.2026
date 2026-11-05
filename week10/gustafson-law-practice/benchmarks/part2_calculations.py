#!/usr/bin/env python3
"""Work out every number used in the Part II files (tasks 06-10).

No extra packages are needed. Exact fractions are used, so there are no
rounding errors until a value is printed. Run it with:
    python3 benchmarks/part2_calculations.py      (Mac / Linux)
    python  benchmarks\\part2_calculations.py      (Windows)
"""
from fractions import Fraction as F


def gustafson(a, p):
    return p - a * (p - 1)


def amdahl(s, p):
    return 1 / (s + (1 - s) / p)


def icbrt(n):
    """Largest whole number x with x**3 <= n."""
    x = round(n ** (1 / 3))
    while x ** 3 > n:
        x -= 1
    while (x + 1) ** 3 <= n:
        x += 1
    return x


# --- Task 06 ---
print("TASK 06: alpha = 0.04")
a = F(4, 100)
print(f"{'P':>6} {'lost = a(P-1)':>14} {'S(P)':>11} {'E(P)':>11} "
      f"{'E - 0.96':>11} {'S if perfect':>13}")
prev = None
for p in [16, 64, 256, 1024, 4096, 16384]:
    s = gustafson(a, p)
    e = s / p
    gap = e - F(96, 100)
    assert s == a + (1 - a) * p and gap == a / p
    print(f"{p:>6} {float(a * (p - 1)):>14.2f} {float(s):>11.2f} "
          f"{float(e):>11.8f} {float(gap):>11.8f} {p:>13}")
    if prev is not None:
        assert prev / gap == 4
    prev = gap

# --- Task 07 ---
print("\nTASK 07: P = 8192, S >= 7500")
p, target = 8192, 7500
a_max = F(p - target, p - 1)
assert gustafson(a_max, p) == target
print("alpha_max =", a_max, "=", f"{float(a_max):.8f}",
      f"({float(a_max) * 100:.4f} %)")
for test in ["0.08448", "0.08449", "0.0845"]:
    print(f"alpha = {test}: S = {float(gustafson(F(test), p)):.3f}")
print("Efficiency needed =", f"{target / p:.6f}")
print(f"Serial time in a 1-hour run = {float(a_max) * 3600:.1f} s")
s_amd = (F(1, target) - F(1, p)) / (1 - F(1, p))
assert amdahl(s_amd, p) == target and s_amd == a_max / target
print(f"Amdahl s needed for the same speedup = {float(s_amd):.4e} "
      f"({float(s_amd) * 100:.5f} %)")

# --- Task 08 ---
print("\nTASK 08: s = alpha = 0.08")
s = F(8, 100)
print("0.92/128 =", float((1 - s) / 128),
      "| bottom =", float(s + (1 - s) / 128))
print(f"{'P':>6} {'Amdahl':>9} {'Gustafson':>10} {'ratio G/A':>10} "
      f"{'% over Amdahl':>14} {'% below Gust.':>14}")
for p in [1, 2, 8, 32, 128, 512, 2048, 16384]:
    x, y = amdahl(s, p), gustafson(s, p)
    r = y / x
    assert r == s**2 + (1 - s)**2 + s * (1 - s) * (p + F(1, p))
    print(f"{p:>6} {float(x):>9.4f} {float(y):>10.2f} {float(r):>10.4f} "
          f"{float((y - x) / x * 100):>14.2f} {float((y - x) / y * 100):>14.2f}")
print("Amdahl ceiling =", float(1 / s),
      "| share of ceiling at 128 =",
      f"{float(amdahl(s, 128) * s) * 100:.2f} %")
print("Fixed job with f = 0.08: alpha(128) =",
      f"{float(s / (s + (1 - s) / 128)):.4f}")
print("Scaled job with alpha = 0.08: f(128) =",
      f"{float(s / gustafson(s, 128)):.6f}")

# --- Task 09 ---
print("\nTASK 09: A = 2%, B = 6%, C = 92%")
A, B, C = F(2, 100), F(6, 100), F(92, 100)
a = A + B
assert A + B + C == 1
print("alpha =", float(a))
for p in [512, 2048]:
    S = gustafson(a, p)
    assert S == a + C * p
    print(f"P={p}: S={float(S):.2f} E={float(S / p):.6f} "
          f"lost={float(a * (p - 1)):.2f} "
          f"(A {float(A * (p - 1)):.2f}, B {float(B * (p - 1)):.2f}) "
          f"| no useful work in A,B: {float(C * p):.2f}")
print("What if (spare time goes to phase C):")
for name, x in [("as measured", a), ("A removed", B), ("B halved", A + B / 2),
                ("B removed", A), ("both halved", a / 2)]:
    print(f"{name:>12}: alpha={float(x):.3f} "
          f"S(512)={float(gustafson(x, 512)):.2f} "
          f"S(2048)={float(gustafson(x, 2048)):.2f}")

# --- Task 10 ---
print("\nTASK 10: 32 GB per node")
for label, gb in [("1 GB = 2^30 bytes", 2**30), ("1 GB = 10^9 bytes", 10**9)]:
    cap = 32 * gb
    print(label)
    for b in [8, 40, 80]:
        n = icbrt(cap // b)
        assert n**3 * b <= cap < (n + 1)**3 * b
        row = [f"  b={b:>2}: block n={n}"]
        for nodes, side in [(64, 4), (512, 8)]:
            total = nodes * cap
            agg = icbrt(total // b)
            assert agg**3 * b <= total < (agg + 1)**3 * b
            row.append(f"{nodes} nodes: total-memory N={agg}, per-node N={n * side}")
        print(" | ".join(row))

cap = 32 * 2**30
print("Total memory, GB:", 64 * 32, 512 * 32)
print("Points per node:", cap // 8, "| side =", f"{(cap / 8) ** (1 / 3):.2f}")
print("1625^3 =", 1625**3, "uses", 1625**3 * 8, "bytes =",
      f"{1625**3 * 8 / 2**30:.3f} GB")
print("1626^3 uses", 1626**3 * 8, "bytes =", f"{1626**3 * 8 / 2**30:.3f} GB",
      "| cap =", cap)
print("N=6500 points:", 6500**3, "| N=13000 points:", 13000**3,
      "| ratio", 13000**3 // 6500**3)
print("Exact real N, 64 nodes:", f"{(64 * cap / 8) ** (1 / 3):.2f}",
      "| 512 nodes:", f"{(512 * cap / 8) ** (1 / 3):.2f}")
print("Memory used at 6500:", f"{6500**3 * 8 / 2**30:.1f} GB of {64 * 32}",
      "| at 13000:", f"{13000**3 * 8 / 2**30:.1f} GB of {512 * 32}")
print(f"Run time factor if time steps grow with N: {13000 / 6500:.0f}")

print("\nAll checks passed.")
