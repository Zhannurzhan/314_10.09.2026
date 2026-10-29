import time, numpy as np
# Allocate flat array in physical memory
data = np.ones(16_777_216, dtype=np.float64)
# Measure read throughput for stride s in [1, 2, 4, 8, 16, 32, 64, 128]
# Run 5 warmup iterations, followed by 5 recorded iterations. Log median time_ns.