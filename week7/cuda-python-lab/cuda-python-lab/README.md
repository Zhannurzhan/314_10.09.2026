# CUDA Python Lab — 50 GPU tasks

```
lab_kernels/   your code: one file per module (Module 1 done, 2-6 are stubs)
tests/         automatic checks (Module 1 written so far)
autograder.py  runs all tests and prints a score per module
colab_launcher.ipynb  open this in Google Colab with a T4 GPU
```

## Workflow
1. Open `colab_launcher.ipynb` in Colab (File → Open notebook → GitHub), set runtime to **T4 GPU**.
2. Run the setup cells.
3. Edit a file in `lab_kernels/`, run its tests, fix, repeat.
4. Push to GitHub after each module.

## Pattern for every task
1. `@cuda.jit` kernel: find my index → bounds check → do one small piece.
2. `run_...` helper: copy to GPU → pick grid/block → launch → copy back.
3. A test comparing against NumPy.

## Testing without a GPU
`NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_module1.py` runs Numba kernels on the CPU (slow; CuPy tasks won't work).
