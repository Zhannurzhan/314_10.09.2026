"""
TASK 1 - Hardware Probe & Serial Baseline
"""

import os

# Force every math library to use ONE thread (strictly serial).
# This must happen BEFORE numpy is imported.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import subprocess
import time

import numpy as np

# ===================== CHANGE THIS =====================
STUDENT_ID = "230109009"  
# =======================================================

N = 20_000_000
OUTPUT_FILE = "task1_output.txt"

lines = []


def log(text=""):
    """Print to the console AND remember the line for task1_output.txt."""
    print(text)
    lines.append(text)


def get_cpu_model():
    """Return the CPU model name on Windows, macOS or Linux."""
    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL).strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def main():
    # Seed must be a whole number, so keep only the digits of the ID.
    digits = "".join(ch for ch in STUDENT_ID if ch.isdigit())
    seed = int(digits) if digits else 0

    # ---- 1. Hardware probe ----
    log("=" * 55)
    log("TASK 1 - Hardware Probe & Serial Baseline")
    log("=" * 55)
    log(f"Student ID       : {STUDENT_ID}")
    log(f"OS               : {platform.system()} {platform.release()} ({platform.version()})")
    log(f"Platform         : {platform.platform()}")
    log(f"CPU model        : {get_cpu_model()}")
    log(f"Architecture     : {platform.machine()}")
    log(f"Logical cores    : {os.cpu_count()}")
    log(f"Python version   : {platform.python_version()}")
    log(f"NumPy version    : {np.__version__}")
    log("-" * 55)

    # ---- 2. Generate data with the Student ID as seed ----
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000
    log(f"Array size       : {N:,} float64 elements (seed = {seed})")
    log(f"Generation time  : {gen_ms:.2f} ms")
    log("-" * 55)

    # ---- 3 & 4. Serial sum + standard deviation, timed ----
    start = time.perf_counter()
    total = np.sum(data)
    std_dev = np.std(data)
    elapsed_ms = (time.perf_counter() - start) * 1000

    log("Mode             : Serial (1 thread)")
    log(f"Sum              : {total:.6f}")
    log(f"Std deviation    : {std_dev:.10f}")
    log(f"Wall-clock time  : {elapsed_ms:.2f} ms")
    log("=" * 55)

    # Save exactly what was printed to the console.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(Output also saved to {OUTPUT_FILE})")


if __name__ == "__main__":
    main()