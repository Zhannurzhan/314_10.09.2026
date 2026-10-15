"""Run all tests and print a per-module score.  Usage:  python autograder.py [module_number]"""
import subprocess, sys

modules = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
for m in modules:
    path = f"tests/test_module{m}.py"
    try:
        open(path).close()
    except FileNotFoundError:
        print(f"Module {m}: no tests yet")
        continue
    r = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "--tb=short"],
                       capture_output=True, text=True)
    summary = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr
    print(f"Module {m}: {summary}")
    if r.returncode != 0:
        print(r.stdout[-3000:])
