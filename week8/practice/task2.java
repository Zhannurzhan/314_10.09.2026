/*
 * TASK 2 — Pseudo-Random Generator State Bottleneck
 *
 * Monte Carlo estimate of PI with 40,000,000 samples on 4 threads:
 *   Mode 1: one SHARED java.util.Random guarded by a ReentrantLock
 *           (every sample = lock -> draw x,y -> unlock, threads queue up).
 *   Mode 2: one INDEPENDENT java.util.Random per thread,
 *           Seed = (Student_ID * 10) + thread_index (no locking at all).
 *
 * Both modes use the same generator class, so the ONLY difference
 * measured is the synchronization on shared generator state.
 *
 * Compile: javac task2.java
 * Run:     java task2
 */
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

public class task2 {

    // ---------------------------------------------------------------
    static final long STUDENT_ID   = 230109009L;   // <-- PUT YOUR STUDENT ID HERE
    static final long TOTAL_SAMPLES = 40_000_000L;
    static final int  THREADS       = 4;
    static final String OUTPUT_FILE = "task2_output.txt";
    // ---------------------------------------------------------------

    static class Result {
        double pi;
        double ms;
        Result(double pi, double ms) { this.pi = pi; this.ms = ms; }
    }

    /* Mode 1: single shared PRNG guarded by a mutex */
    static Result runShared(long totalSamples) throws InterruptedException {
        final Random shared = new Random(STUDENT_ID * 10);
        final ReentrantLock lock = new ReentrantLock();
        final long[] hits = new long[THREADS];
        final long perThread = totalSamples / THREADS;
        Thread[] workers = new Thread[THREADS];

        long t0 = System.nanoTime();
        for (int t = 0; t < THREADS; t++) {
            final int id = t;
            final long n = (t == THREADS - 1) ? totalSamples - perThread * (THREADS - 1) : perThread;
            workers[t] = new Thread(() -> {
                long local = 0;
                for (long i = 0; i < n; i++) {
                    double x, y;
                    lock.lock();
                    try {
                        x = shared.nextDouble();
                        y = shared.nextDouble();
                    } finally {
                        lock.unlock();
                    }
                    if (x * x + y * y <= 1.0) local++;
                }
                hits[id] = local;
            });
            workers[t].start();
        }
        for (Thread w : workers) w.join();
        long t1 = System.nanoTime();

        long totalHits = 0;
        for (long h : hits) totalHits += h;
        return new Result(4.0 * totalHits / totalSamples, (t1 - t0) / 1e6);
    }

    /* Mode 2: independent PRNG per thread, Seed = (Student_ID * 10) + thread_index */
    static Result runIndependent(long totalSamples) throws InterruptedException {
        final long[] hits = new long[THREADS];
        final long perThread = totalSamples / THREADS;
        Thread[] workers = new Thread[THREADS];

        long t0 = System.nanoTime();
        for (int t = 0; t < THREADS; t++) {
            final int id = t;
            final long n = (t == THREADS - 1) ? totalSamples - perThread * (THREADS - 1) : perThread;
            workers[t] = new Thread(() -> {
                Random rng = new Random((STUDENT_ID * 10) + id);
                long local = 0;
                for (long i = 0; i < n; i++) {
                    double x = rng.nextDouble();
                    double y = rng.nextDouble();
                    if (x * x + y * y <= 1.0) local++;
                }
                hits[id] = local;
            });
            workers[t].start();
        }
        for (Thread w : workers) w.join();
        long t1 = System.nanoTime();

        long totalHits = 0;
        for (long h : hits) totalHits += h;
        return new Result(4.0 * totalHits / totalSamples, (t1 - t0) / 1e6);
    }

    public static void main(String[] args) throws Exception {
        // JIT warm-up on a small run so compilation time isn't measured
        runShared(400_000);
        runIndependent(400_000);

        Result shared = runShared(TOTAL_SAMPLES);
        Result indep  = runIndependent(TOTAL_SAMPLES);
        double speedup = shared.ms / indep.ms;

        StringBuilder sb = new StringBuilder();
        String line = "=".repeat(60);
        String thin = "-".repeat(60);
        sb.append(line).append('\n');
        sb.append(" TASK 2 - PRNG State Bottleneck (Monte Carlo PI)\n");
        sb.append(line).append('\n');
        sb.append(String.format(" Student ID          : %d%n", STUDENT_ID));
        sb.append(String.format(" Logical cores       : %d%n", Runtime.getRuntime().availableProcessors()));
        sb.append(String.format(" Threads             : %d%n", THREADS));
        sb.append(String.format(" Samples             : %,d%n", TOTAL_SAMPLES));
        sb.append(thin).append('\n');
        sb.append(" Shared PRNG (mutex-guarded)\n");
        sb.append(String.format("   Computed PI       : %.10f%n", shared.pi));
        sb.append(String.format("   Absolute error    : %.10f%n", Math.abs(shared.pi - Math.PI)));
        sb.append(String.format("   Wall time         : %.2f ms%n", shared.ms));
        sb.append(thin).append('\n');
        sb.append(" Independent PRNG (Seed = ID*10 + thread_index)\n");
        sb.append(String.format("   Computed PI       : %.10f%n", indep.pi));
        sb.append(String.format("   Absolute error    : %.10f%n", Math.abs(indep.pi - Math.PI)));
        sb.append(String.format("   Wall time         : %.2f ms%n", indep.ms));
        sb.append(thin).append('\n');
        sb.append(String.format(" Speedup (Shared / Independent) : %.2fx%n", speedup));
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