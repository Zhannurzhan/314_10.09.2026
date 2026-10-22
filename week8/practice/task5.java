/*
 * TASK 5 — Thread Oversubscription Threshold
 *
 * 10,000 small CPU-bound tasks; each one hashes a 1 KB buffer with SHA-256
 * 500 times (the digest is fed back into the buffer so nothing can be skipped).
 * The whole batch is run on thread pools of size
 *     [1, Cores/2, Cores, Cores*2, Cores*4, Cores*16, 256]
 * and the wall-clock time is the median of RUNS repetitions.
 *
 * Compile: javac task5.java
 * Run:     java task5
 */
import java.io.FileWriter;
import java.io.IOException;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class task5 {

    // ---------------------------------------------------------------
    static final long STUDENT_ID   = 230109009L;   // <-- PUT YOUR STUDENT ID HERE
    static final int  TASKS        = 10_000;
    static final int  HASHES_PER_TASK = 500;
    static final int  PAYLOAD_BYTES   = 1024;
    static final int  RUNS         = 3;           // repetitions per pool size (median)
    static final String OUTPUT_FILE = "task5_output.txt";
    // ---------------------------------------------------------------

    static final int CORES = Runtime.getRuntime().availableProcessors();

    /* One small CPU-bound task: hash a 1 KB string 500 times */
    static long hashTask(int taskId) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] buf = new byte[PAYLOAD_BYTES];
        new Random(STUDENT_ID * 31 + taskId).nextBytes(buf);
        byte[] d = null;
        for (int i = 0; i < HASHES_PER_TASK; i++) {
            d = md.digest(buf);
            System.arraycopy(d, 0, buf, 0, d.length);   // chain: next hash depends on this one
        }
        return ((d[0] & 0xffL) << 8) | (d[1] & 0xffL);
    }

    /* Run the full batch on a pool of the given size, return ms */
    static double runBatch(int poolSize, int tasks) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(poolSize);
        List<Callable<Long>> jobs = new ArrayList<>(tasks);
        for (int t = 0; t < tasks; t++) {
            final int id = t;
            jobs.add(() -> hashTask(id));
        }
        long t0 = System.nanoTime();
        long checksum = 0;
        for (Future<Long> f : pool.invokeAll(jobs)) checksum += f.get();
        long t1 = System.nanoTime();
        pool.shutdown();
        if (checksum < 0) System.out.print("");      // keep result alive
        return (t1 - t0) / 1e6;
    }

    public static void main(String[] args) throws Exception {
        // Pool sizes as specified; duplicates removed (e.g. Cores/2 == 1 on 2 cores)
        int[] requested = {1, Math.max(1, CORES / 2), CORES, CORES * 2, CORES * 4, CORES * 16, 256};
        TreeSet<Integer> uniq = new TreeSet<>();
        for (int p : requested) uniq.add(p);
        int[] sizes = uniq.stream().mapToInt(Integer::intValue).toArray();

        // JIT warm-up (not timed)
        runBatch(CORES, 500);

        double[] times = new double[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            double[] r = new double[RUNS];
            for (int k = 0; k < RUNS; k++) r[k] = runBatch(sizes[i], TASKS);
            Arrays.sort(r);
            times[i] = r[RUNS / 2];
            System.err.printf("  measured pool=%d -> %.1f ms%n", sizes[i], times[i]);
        }

        // Find optimum and the threshold where time starts worsening
        int best = 0;
        for (int i = 1; i < times.length; i++) if (times[i] < times[best]) best = i;
        int threshold = (best + 1 < sizes.length) ? best + 1 : -1;

        String line = "=".repeat(86);
        String thin = "-".repeat(86);
        StringBuilder sb = new StringBuilder();
        sb.append(line).append('\n');
        sb.append(" TASK 5 - Thread Oversubscription Threshold\n");
        sb.append(line).append('\n');
        sb.append(String.format(" Student ID: %d | Logical cores: %d | Tasks: %,d x %d SHA-256 of %d B | Runs: median of %d%n",
                STUDENT_ID, CORES, TASKS, HASHES_PER_TASK, PAYLOAD_BYTES, RUNS));
        sb.append(thin).append('\n');
        sb.append(String.format(" %-10s | %-20s | %-29s | %s%n",
                "Pool Size", "Execution Time (ms)", "Speedup relative to 1 thread", "Status"));
        sb.append(thin).append('\n');
        for (int i = 0; i < sizes.length; i++) {
            String status;
            if (i == best)               status = "OPTIMAL (fastest)";
            else if (i == threshold)     status = "<-- THRESHOLD: worsening starts (context switching)";
            else if (i < best)           status = (sizes[i] == 1) ? "Baseline" : "Scaling";
            else                         status = "Oversubscribed";
            sb.append(String.format(" %-10d | %20.2f | %28.2fx | %s%n",
                    sizes[i], times[i], times[0] / times[i], status));
        }
        sb.append(thin).append('\n');
        if (threshold >= 0) {
            sb.append(String.format(" Degradation threshold: pool size %d (%.1f%% slower than optimum at pool size %d)%n",
                    sizes[threshold], (times[threshold] / times[best] - 1) * 100, sizes[best]));
        } else {
            sb.append(" No degradation observed up to the largest pool size tested.\n");
        }
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