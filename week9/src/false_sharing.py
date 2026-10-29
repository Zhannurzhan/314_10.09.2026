import multiprocessing as mp, ctypes, time
def worker_unpadded(counter_array, worker_id, iterations):
 # Pathological: counter_array is a contiguous array of 64-bit integers.
 # worker 0 writes to index 0; worker 1 writes to index 1.
 # Indices 0 through 7 reside on the EXACT SAME 64-byte physical cache line!
   for _ in range(iterations):
        counter_array[worker_id] += 1