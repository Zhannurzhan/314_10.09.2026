/*
 * TASK 3 — Prefix Scan (Step Efficiency vs. Work Efficiency)
 *
 * Serial inclusive scan:   out[i] = out[i-1] + in[i]        (n additions, n steps)
 * Parallel 2-pass scan:
 *   Pass 1 (parallel): each worker sums its own chunk          -> chunkSum[t]
 *   Offset reduction (serial, tiny): exclusive scan of chunkSum -> offset[t]
 *   Pass 2 (parallel): each worker scans its chunk starting at offset[t]
 * The parallel version does ~2n additions (less work-efficient) but in
 * ~2n/P steps per thread (more step-efficient).
 *
 * Compile: javac task3.java
 * Run:     java task3
 * Exit code 1 if parallel_output[-1] != serial_output[-1].
 */
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class task3 {

    // ---------------------------------------------------------------
    static final long STUDENT_ID  = 230109009L;   // <-- PUT YOUR STUDENT ID HERE
    static final int  N           = 2_000_000;
    static final int  RUNS        = 11;          // timed repetitions (median reported)
    static final String OUTPUT_FILE = "task3_output.txt";
    // ---------------------------------------------------------------

    static final int THREADS = Runtime.getRuntime().availableProcessors();
    static final ExecutorService POOL = Executors.newFixedThreadPool(THREADS);

    /* Serial inclusive prefix sum */
    static void serialScan(int[] in, long[] out) {
        long running = 0;
        for (int i = 0; i < in.length; i++) {
            running += in[i];
            out[i] = running;
        }
    }

    /* Parallel chunked 2-pass inclusive prefix scan */
    static void parallelScan(int[] in, long[] out) throws Exception {
        final int n = in.length;
        final int chunk = (n + THREADS - 1) / THREADS;
        final long[] chunkSum = new long[THREADS];

        // Pass 1: local chunk sums (parallel)
        List<Callable<Void>> pass1 = new ArrayList<>();
        for (int t = 0; t < THREADS; t++) {
            final int id = t;
            pass1.add(() -> {
                int start = id * chunk, end = Math.min(start + chunk, n);
                long s = 0;
                for (int i = start; i < end; i++) s += in[i];
                chunkSum[id] = s;
                return null;
            });
        }
        POOL.invokeAll(pass1);

        // Offset reduction: exclusive scan of chunk sums (serial, P elements)
        final long[] offset = new long[THREADS];
        long acc = 0;
        for (int t = 0; t < THREADS; t++) {
            offset[t] = acc;
            acc += chunkSum[t];
        }

        // Pass 2: apply offsets while scanning each chunk (parallel)
        List<Callable<Void>> pass2 = new ArrayList<>();
        for (int t = 0; t < THREADS; t++) {
            final int id = t;
            pass2.add(() -> {
                int start = id * chunk, end = Math.min(start + chunk, n);
                long running = offset[id];
                for (int i = start; i < end; i++) {
                    running += in[i];
                    out[i] = running;
                }
                return null;
            });
        }
        POOL.invokeAll(pass2);
    }

    static double median(double[] a) {
        double[] c = a.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        // 1. Generate 2,000,000 integers in [1, 10] from the Student ID seed
        Random rng = new Random(STUDENT_ID);
        int[] input = new int[N];
        for (int i = 0; i < N; i++) input[i] = rng.nextInt(10) + 1;

        long[] serialOut = new long[N];
        long[] parallelOut = new long[N];

        // JIT warm-up (not timed)
        for (int w = 0; w < 5; w++) {
            serialScan(input, serialOut);
            parallelScan(input, parallelOut);
        }

        // 2. Benchmark
        double[] st = new double[RUNS], pt = new double[RUNS];
        for (int r = 0; r < RUNS; r++) {
            Arrays.fill(serialOut, 0);
            long t0 = System.nanoTime();
            serialScan(input, serialOut);
            st[r] = (System.nanoTime() - t0) / 1e6;

            Arrays.fill(parallelOut, 0);
            t0 = System.nanoTime();
            parallelScan(input, parallelOut);
            pt[r] = (System.nanoTime() - t0) / 1e6;
        }
        POOL.shutdown();

        double serialMs = median(st), parallelMs = median(pt);
        double speedup = serialMs / parallelMs;

        // 3. Verification constraint: parallel_output[-1] == serial_output[-1]
        boolean lastMatch = parallelOut[N - 1] == serialOut[N - 1];
        boolean fullMatch = Arrays.equals(parallelOut, serialOut);
        String status = lastMatch ? "PASS" : "FAIL";

        // 4. Report
        String line = "=".repeat(78);
        StringBuilder sb = new StringBuilder();
        sb.append(line).append('\n');
        sb.append(" TASK 3 - Parallel Prefix Scan (Step vs Work Efficiency)\n");
        sb.append(line).append('\n');
        sb.append(String.format(" Student ID: %d | Threads: %d | N: %,d | Runs (median): %d%n",
                STUDENT_ID, THREADS, N, RUNS));
        sb.append(String.format(" serial_output[-1] = %d | parallel_output[-1] = %d | full array match: %b%n",
                serialOut[N - 1], parallelOut[N - 1], fullMatch));
        sb.append("-".repeat(78)).append('\n');
        sb.append(String.format(" Serial Time: %.3f ms | Parallel Time: %.3f ms | Speedup: %.2fx | Verification: %s%n",
                serialMs, parallelMs, speedup, status));
        sb.append(line).append('\n');

        String report = sb.toString();
        System.out.print(report);
        try (FileWriter fw = new FileWriter(OUTPUT_FILE)) {
            fw.write(report);
        } catch (IOException e) {
            System.err.println("Could not write " + OUTPUT_FILE + ": " + e.getMessage());
        }

        if (!lastMatch) {
            System.err.println("ASSERTION FAILED: parallel_output[-1] != serial_output[-1]");
            System.exit(1);
        }
    }
}