/*
 * TASK 6 — Atomic CAS vs. OS Mutex Under Heavy Contention
 *
 * 8 threads x 1,000,000 increments on ONE shared counter (starts at 0).
 *   Primitive 1: ReentrantLock      -> lock(); counter++; unlock();
 *   Primitive 2: AtomicInteger CAS  -> explicit compareAndSet() retry loop
 * All threads are released together by a latch so contention is maximal.
 * Each primitive is timed as the median of RUNS repetitions, and the final
 * counter is checked to be exactly 8,000,000 every run.
 *
 * Compile: javac task6.java
 * Run:     java task6
 */
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

public class task6 {

    // ---------------------------------------------------------------
    static final long STUDENT_ID = 230109009L;   // <-- PUT YOUR STUDENT ID HERE
    static final int  THREADS    = 8;
    static final int  INCREMENTS = 1_000_000;   // per thread
    static final int  RUNS       = 5;           // median of RUNS
    static final String OUTPUT_FILE = "task6_output.txt";
    // ---------------------------------------------------------------

    static final long TOTAL_OPS = (long) THREADS * INCREMENTS;

    /* ---------- Primitive 1: mutex ---------- */
    static int mutexCounter;

    static double runMutex() throws InterruptedException {
        mutexCounter = 0;
        final ReentrantLock lock = new ReentrantLock();
        final CountDownLatch start = new CountDownLatch(1);
        Thread[] ts = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            ts[t] = new Thread(() -> {
                try { start.await(); } catch (InterruptedException e) { return; }
                for (int i = 0; i < INCREMENTS; i++) {
                    lock.lock();
                    try {
                        mutexCounter++;
                    } finally {
                        lock.unlock();
                    }
                }
            });
            ts[t].start();
        }
        long t0 = System.nanoTime();
        start.countDown();
        for (Thread th : ts) th.join();
        long t1 = System.nanoTime();
        if (mutexCounter != TOTAL_OPS)
            throw new IllegalStateException("Mutex counter wrong: " + mutexCounter);
        return (t1 - t0) / 1e6;
    }

    /* ---------- Primitive 2: hardware atomic (AtomicInteger) ----------
     * incrementAndGet() is a single hardware atomic read-modify-write
     * (LOCK XADD on x86, LSE atomic / CAS on ARM). No OS involvement. */
    static double runAtomic() throws InterruptedException {
        final AtomicInteger counter = new AtomicInteger(0);
        final CountDownLatch start = new CountDownLatch(1);
        Thread[] ts = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            ts[t] = new Thread(() -> {
                try { start.await(); } catch (InterruptedException e) { return; }
                for (int i = 0; i < INCREMENTS; i++) counter.incrementAndGet();
            });
            ts[t].start();
        }
        long t0 = System.nanoTime();
        start.countDown();
        for (Thread th : ts) th.join();
        long t1 = System.nanoTime();
        if (counter.get() != TOTAL_OPS)
            throw new IllegalStateException("Atomic counter wrong: " + counter.get());
        return (t1 - t0) / 1e6;
    }

    /* ---------- Supplementary: explicit software CAS retry loop ---------- */
    static final LongAdder casRetries = new LongAdder();

    static double runCas() throws InterruptedException {
        final AtomicInteger counter = new AtomicInteger(0);
        final CountDownLatch start = new CountDownLatch(1);
        casRetries.reset();
        Thread[] ts = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            ts[t] = new Thread(() -> {
                try { start.await(); } catch (InterruptedException e) { return; }
                long retries = 0;
                for (int i = 0; i < INCREMENTS; i++) {
                    while (true) {
                        int cur = counter.get();                     // read current value
                        if (counter.compareAndSet(cur, cur + 1))     // CAS: swap only if unchanged
                            break;
                        retries++;                                   // another thread won; retry
                    }
                }
                casRetries.add(retries);
            });
            ts[t].start();
        }
        long t0 = System.nanoTime();
        start.countDown();
        for (Thread th : ts) th.join();
        long t1 = System.nanoTime();
        if (counter.get() != TOTAL_OPS)
            throw new IllegalStateException("CAS counter wrong: " + counter.get());
        return (t1 - t0) / 1e6;
    }

    static double median(double[] a) {
        double[] c = a.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        // JIT warm-up (not timed)
        runMutex();
        runAtomic();
        runCas();

        double[] m = new double[RUNS], a = new double[RUNS], c = new double[RUNS];
        long retriesLastRun = 0;
        for (int r = 0; r < RUNS; r++) {
            m[r] = runMutex();
            a[r] = runAtomic();
            c[r] = runCas();
            retriesLastRun = casRetries.sum();
        }
        double mutexMs = median(m), atomicMs = median(a), casMs = median(c);
        double mutexTput  = TOTAL_OPS / (mutexMs / 1000.0);
        double atomicTput = TOTAL_OPS / (atomicMs / 1000.0);
        double casTput    = TOTAL_OPS / (casMs / 1000.0);
        double ratio      = atomicTput / mutexTput;
        double loopRatio  = casTput / mutexTput;

        String line = "=".repeat(64);
        String thin = "-".repeat(64);
        StringBuilder sb = new StringBuilder();
        sb.append(line).append('\n');
        sb.append(" TASK 6 - Atomic CAS vs. Mutex Under Heavy Contention\n");
        sb.append(line).append('\n');
        sb.append(String.format(" Student ID      : %d%n", STUDENT_ID));
        sb.append(String.format(" Logical cores   : %d%n", Runtime.getRuntime().availableProcessors()));
        sb.append(String.format(" Threads         : %d x %,d increments = %,d ops%n", THREADS, INCREMENTS, TOTAL_OPS));
        sb.append(String.format(" Final counter   : %,d (verified for every primitive, every run)%n", TOTAL_OPS));
        sb.append(String.format(" Timing          : median of %d runs%n", RUNS));
        sb.append(thin).append('\n');
        sb.append(" Primitive 1: Mutex (ReentrantLock)\n");
        sb.append(String.format("   Execution time : %10.2f ms%n", mutexMs));
        sb.append(String.format("   Throughput     : %,15.0f ops/sec%n", mutexTput));
        sb.append(thin).append('\n');
        sb.append(" Primitive 2: Atomic CAS (AtomicInteger.incrementAndGet)\n");
        sb.append(String.format("   Execution time : %10.2f ms%n", atomicMs));
        sb.append(String.format("   Throughput     : %,15.0f ops/sec%n", atomicTput));
        sb.append(thin).append('\n');
        sb.append(String.format(" Throughput Ratio (CAS / Mutex) : %.2fx%n", ratio));
        sb.append(thin).append('\n');
        sb.append(" Supplementary: explicit get()+compareAndSet() retry loop\n");
        sb.append(String.format("   Execution time : %10.2f ms%n", casMs));
        sb.append(String.format("   Throughput     : %,15.0f ops/sec%n", casTput));
        sb.append(String.format("   Failed CAS retries (last run): %,d%n", retriesLastRun));
        sb.append(String.format("   Ratio vs Mutex : %.2fx%n", loopRatio));
        sb.append(line).append('\n');

        String report = sb.toString();
        System.out.print(report);
        try (FileWriter fw = new FileWriter(OUTPUT_FILE)) {
            fw.write(report);
        } catch (IOException e) {
            System.err.println("Could not write " + OUTPUT_FILE + ": " + e.getMessage());
        }
    }
}