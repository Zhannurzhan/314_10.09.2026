#!/usr/bin/env python3
"""Work out every number used in the Part III files (tasks 11-15).

No extra packages are needed. Run it with:
    python3 benchmarks/part3_calculations.py      (Mac / Linux)
    python  benchmarks\\part3_calculations.py      (Windows)
"""
from math import log, log2, sqrt


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol * max(1, abs(x), abs(y))


def gustafson(a, p):
    return p - a * (p - 1)


def amdahl(s, p):
    return 1 / (s + (1 - s) / p)


# --- Task 11: Sun-Ni memory-bounded speedup ---
def sun_ni(f, n, g):
    return (f + (1 - f) * g) / (f + (1 - f) * g / n)


print("TASK 11: f = 0.04")
f = 0.04
for n in [1, 16, 1024, 65536]:
    assert close(sun_ni(f, n, 1), amdahl(f, n))
    assert close(sun_ni(f, n, n), gustafson(f, n))
print(f"{'n':>6} | {'G=1':>8} {'time':>6} | {'G=n':>9} {'time':>6} | "
      f"{'G=n^1.5':>9} {'time':>8} | {'G=n^(4/3)':>9} {'time':>7} | "
      f"{'G=sqrt n':>8} {'time':>6}")
for n in [16, 64, 256, 1024, 4096, 16384]:
    parts = []
    for g in [1, n, n**1.5, n**(4 / 3), sqrt(n)]:
        parts.append((sun_ni(f, n, g), f + (1 - f) * g / n))
    print(f"{n:>6} | {parts[0][0]:>8.2f} {parts[0][1]:>6.3f} | "
          f"{parts[1][0]:>9.2f} {parts[1][1]:>6.2f} | "
          f"{parts[2][0]:>9.2f} {parts[2][1]:>8.2f} | "
          f"{parts[3][0]:>9.2f} {parts[3][1]:>7.2f} | "
          f"{parts[4][0]:>8.2f} {parts[4][1]:>6.3f}")
n = 1024
prev = 0
for g in [1, sqrt(n), n, n**(4 / 3), n**1.5, 1e12]:
    cur = sun_ni(f, n, g)
    assert prev < cur < n
    prev = cur
print("Speedup rises with G and stays below n: checked")
print("Largest possible underestimate 1/(1-f) =", f"{1 / (1 - f):.4f}")
print("Serial share of the run itself at n=1024, G=n^1.5:",
      f"{f / (f + (1 - f) * n**0.5):.5f}")
print("Work done (relative) at n=1024, G=n^1.5:",
      f"{f + (1 - f) * n**1.5:.2f}")
print("Fixed-time matrix product: N grows", f"{n**(1 / 3):.2f}x,",
      "memory per node", f"{n**(2 / 3) / n:.4f} of base at n=1024")
print("Memory-bound matrix product: N grows", f"{sqrt(n):.0f}x")

# --- Task 12: communication penalty ---
print("\nTASK 12: alpha = 0.04")
a = 0.04


def s_time_grows(p, c):
    return (a + (1 - a) * p) / (1 + c)


def s_fixed_time(p, c):
    return a + (1 - a - c) * p


print(f"{'P':>6} {'classic':>10} | {'C log':>7} {'grow S':>9} {'E':>6} "
      f"{'fix S':>9} | {'C sqrt':>7} {'grow S':>9} {'E':>6} {'fix S':>9}")
for p in [1, 16, 256, 1024, 4096, 16384, 65536, 262144]:
    c1 = 0.005 * log2(p)
    c2 = 0.001 * (sqrt(p) - 1)
    g = gustafson(a, p)
    assert close(s_fixed_time(p, c1), g - c1 * p)
    print(f"{p:>6} {g:>10.2f} | {c1:>7.3f} {s_time_grows(p, c1):>9.2f} "
          f"{s_time_grows(p, c1) / p:>6.3f} {s_fixed_time(p, c1):>9.2f} | "
          f"{c2:>7.3f} {s_time_grows(p, c2):>9.2f} "
          f"{s_time_grows(p, c2) / p:>6.3f} {s_fixed_time(p, c2):>9.2f}")

# Strict fixed time with sqrt overhead: where is the peak and the collapse?
c0 = 0.001
peak_root = (1 - a + c0) / (1.5 * c0)
peak_p = peak_root ** 2
zero_root = (1 - a) / c0 + 1
print(f"sqrt overhead c=0.001: peak at P = {peak_p:.0f}, "
      f"S = {s_fixed_time(peak_p, c0 * (sqrt(peak_p) - 1)):.0f}; "
      f"parallel share gone at P = {zero_root**2:.0f}")
best = max(range(1000, 1000000, 500),
           key=lambda q: s_fixed_time(q, c0 * (sqrt(q) - 1)))
assert abs(best - peak_p) < 1000
print(f"log overhead c=0.005: peak near log2(P) = "
      f"{(1 - a - 0.005 / log(2)) / 0.005:.1f}")
# Linear overhead brings back a ceiling.
c3 = 0.001
print("Linear overhead c=0.001, time-grows model:")
for p in [1024, 65536, 2**24]:
    print(f"  P={p}: S = {s_time_grows(p, c3 * (p - 1)):.1f}")
print("  ceiling (1-a)/c =", round((1 - a) / c3, 1))
for p, c in [(1024, 0.005 * 10), (1024, 0.001 * (sqrt(1024) - 1))]:
    print(f"alpha_eff at P={p}, C={c:.3f}: {(a + c) / (1 + c):.4f}")

# --- Task 13: failure cases ---
print("\nTASK 13A: CG on a 3D grid, per-node block fixed")
print(f"{'P':>7} {'iter factor':>12} {'per-iter':>9} {'time':>8} {'eff':>7}")
for p in [1, 8, 64, 512, 4096, 32768, 262144]:
    k = p ** (1 / 3)
    it = 1 + 2 * 0.01 * log2(p)
    t = k * it
    print(f"{p:>7} {k:>12.0f} {it:>9.2f} {t:>8.2f} {1 / t:>7.3f}")

print("\nTASK 13B: central queue, w = 1000 us, q = 10 us")
w, q = 1000.0, 10.0
sat = (w + q) / q
print("Saturation at P =", sat)
print(f"{'P':>7} {'time per k tasks':>17} {'speedup':>9} {'eff':>7} {'queue busy':>11}")
for p in [1, 16, 64, 101, 256, 1024, 4096]:
    t = max(w + q, p * q)
    sp = p * (w + q) / t
    assert close(sp, min(p, sat))
    print(f"{p:>7} {t:>17.0f} {sp:>9.1f} {sp / p:>7.3f} {p * q / t:>11.3f}")
for p in [1024, 4096]:
    print(f"Time factor at P={p}: {max(w + q, p * q) / (w + q):.2f}")
print("Ceiling compared with Amdahl with f = q/(w+q):",
      f"{1 / (q / (w + q)):.1f}")

# --- Task 14: isoefficiency ---
print("\nTASK 14")
a = 0.04
for p in [2, 16, 1024, 65536]:
    w_total = gustafson(a, p)
    to = p * 1 - w_total
    assert close(to, a * (p - 1))
    assert close(w_total / p, 1 / (1 + to / w_total))
    assert close(w_total, 1 + (1 - a) * (p - 1))
print("To = alpha(P-1) and W = 1 + (1-alpha)(P-1): checked")
s0 = 1.0
print("Fixed target E = 0.96 with K = 24 (alpha = 0.04):")
for p in [16, 1024, 65536]:
    k = 0.96 / 0.04
    w_iso = k * 0.04 * (p - 1)
    assert close(w_iso / (w_iso + 0.04 * (p - 1)), 0.96)
    print(f"  P={p}: W_iso = {w_iso:.2f}, Gustafson W = {gustafson(a, p):.2f}")
print("Efficiency for grain size w with serial time s0 = 1, P = 1024:")
for g in [1, 4, 24, 99]:
    e = (1 + g * 1024) / (1024 * (1 + g))
    print(f"  w={g:>3}: E = {e:.4f}, limit = {g / (1 + g):.4f}")
for e in [0.5, 0.8, 0.9, 0.96, 0.99]:
    print(f"  E={e}: K = {e / (1 - e):.0f}")

# --- Task 15: energy ---
print("\nTASK 15: alpha = 0.04, Pidle = 50 W, Pactive = 100 W, T = 1 hour")
a, pi, pa, t = 0.04, 50.0, 100.0, 1.0
r = pi / pa
e1 = t * (pi + pa)
print("E(1) =", e1, "Wh")
print(f"{'P':>6} {'S(P)':>10} {'E(P) kWh':>10} {'avg kW':>8} {'E/E(1)':>9} "
      f"{'S ratio':>9} {'per-work':>9} | {'r=0':>6} {'r=2':>6}")
for p in [1, 16, 256, 1024, 4096, 16384]:
    s = gustafson(a, p)
    e = t * (p * pi + s * pa)
    per = (e / s) / e1
    eff = s / p
    assert close(per, (r / eff + 1) / (r + 1))
    print(f"{p:>6} {s:>10.2f} {e / 1000:>10.2f} {e / t / 1000:>8.2f} "
          f"{e / e1:>9.2f} {s:>9.2f} {per:>9.4f} | "
          f"{(0 / eff + 1) / (0 + 1):>6.3f} {(2 / eff + 1) / 3:>6.3f}")
print("Limit of per-work ratio:",
      f"{(r / (1 - a) + 1) / (r + 1):.4f}")
print("Amdahl fixed job, s = 0.04, energy per job compared with 1 core:")
for p in [16, 256, 1024, 4096, 16384]:
    eff = amdahl(0.04, p) / p
    print(f"  P={p}: efficiency {eff:.4f}, energy ratio "
          f"{(r / eff + 1) / (r + 1):.2f}")
p = 1024
s = gustafson(a, p)
print(f"Same scaled job on one core: time {s:.2f} h, "
      f"energy {s * e1 / 1000:.2f} kWh; on 1024: "
      f"{(p * pi + s * pa) / 1000:.2f} kWh, 1 h")

extra = (p * pi + s * pa) - s * e1
assert close(extra, t * pi * a * (p - 1))
print(f"Extra energy from idle processors: {extra:.0f} Wh")
idle_on = p * pi + pa
print(f"One busy node with 1023 idle nodes left on: {idle_on:.0f} Wh, "
      f"{idle_on / ((p * pi + s * pa) / s):.1f} times the energy per unit "
      f"of work of the full run")

print("\nAll checks passed.")
