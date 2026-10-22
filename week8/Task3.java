/*
 * TASK 3 - Cache Line Contention (False Sharing)
 *
 * Run:  java Task3.java          (uses your physical core count)
 *       java Task3.java 8        (force 8 threads)
 */
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLongArray;

public class Task3 {

    // ===================== CHANGE THIS =====================
    static final String STUDENT_ID = "230109009";
    // =======================================================

    static final long ITERATIONS = 50_000_000L;  // increments per thread
    static final int PAD = 16;                   // 16 longs x 8 bytes = 128 bytes apart
    static final int RUNS = 5;                   // each version runs 5 times, median kept
    static final String OUTPUT_FILE = "task3_output.txt";

    static final StringBuilder output = new StringBuilder();

    /** Print to the console AND remember the line for task3_output.txt. */
    static void log(String text) {
        System.out.println(text);
        output.append(text).append("\n");
    }

    /** Ask the operating system how many PHYSICAL cores the CPU has. */
    static int physicalCores() {
        String os = System.getProperty("os.name").toLowerCase();
        String[] cmd;
        if (os.contains("win")) {
            cmd = new String[]{"powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"};
        } else if (os.contains("mac")) {
            cmd = new String[]{"sysctl", "-n", "hw.physicalcpu"};
        } else {
            cmd = new String[]{"sh", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"};
        }
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String text = new String(p.getInputStream().readAllBytes()).trim();
            p.waitFor();
            int n = Integer.parseInt(text.split("\\s+")[0]);
            if (n > 0) return n;
        } catch (Exception ignored) { }
        // Fallback: assume 2 logical threads per physical core
        return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    }

    interface Work { void run(int id); }

    /** Start n threads running the same work, wait for all, return time in ms. */
    static double timeThreads(int n, Work work) throws InterruptedException {
        Thread[] threads = new Thread[n];
        for (int i = 0; i < n; i++) {
            final int id = i;
            threads[i] = new Thread(() -> work.run(id));
        }
        long start = System.nanoTime();
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();
        return (System.nanoTime() - start) / 1_000_000.0;
    }

    // The three kinds of work. They are separate methods so the warm-up
    // compiles them once and the timed runs reuse the fast compiled code.
    static void unpaddedWork(AtomicLongArray arr, int id, long iters) {
        for (long k = 0; k < iters; k++) arr.incrementAndGet(id);          // neighbours: 8 bytes apart
    }

    static void paddedWork(AtomicLongArray arr, int id, long iters) {
        for (long k = 0; k < iters; k++) arr.incrementAndGet(id * PAD);    // 128 bytes apart
    }

    static void localWork(AtomicLongArray arr, int id, long iters) {
        long count = 0;
        for (long k = 0; k < iters; k++) count++;                          // stays in a CPU register
        arr.set(id, count);                                                // one shared write at the end
    }

    static double median(double[] values) {
        double[] v = values.clone();
        Arrays.sort(v);
        return v[v.length / 2];
    }

    static void check(String name, AtomicLongArray arr, int n, int stride) {
        for (int i = 0; i < n; i++) {
            if (arr.get(i * stride) != ITERATIONS) {
                throw new IllegalStateException(name + ": counter " + i + " = " + arr.get(i * stride));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int logical = Runtime.getRuntime().availableProcessors();
        int physical = physicalCores();
        int n = (args.length > 0) ? Integer.parseInt(args[0]) : physical;

        log("=".repeat(62));
        log("TASK 3 - Cache Line Contention (False Sharing)");
        log("=".repeat(62));
        log("Student ID         : " + STUDENT_ID);
        log("OS                 : " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        log("Java version       : " + System.getProperty("java.version"));
        log("Logical cores      : " + logical);
        log("Physical cores     : " + physical);
        log("Threads used (N)   : " + n);
        log(String.format("Increments/thread  : %,d", ITERATIONS));
        log("Runs per version   : " + RUNS + " (median reported)");
        log("-".repeat(62));

        double[] tShared = new double[RUNS];
        double[] tPadded = new double[RUNS];
        double[] tLocal  = new double[RUNS];

        // Warm-up (not timed): lets the Java JIT compile the loops first.
        long warm = ITERATIONS / 10;
        timeThreads(n, id -> unpaddedWork(new AtomicLongArray(n), id, warm));
        AtomicLongArray w1 = new AtomicLongArray(n), w2 = new AtomicLongArray(n * PAD), w3 = new AtomicLongArray(n);
        timeThreads(n, id -> unpaddedWork(w1, id, warm));
        timeThreads(n, id -> paddedWork(w2, id, warm));
        timeThreads(n, id -> localWork(w3, id, warm));

        for (int r = 0; r < RUNS; r++) {
            // ---- Version A: UNPADDED shared array ----
            // Counters sit next to each other (8 bytes apart), so they share
            // ONE 64-byte cache line. Every write by one core invalidates that
            // line in every other core's cache -> the line "ping-pongs".
            AtomicLongArray shared = new AtomicLongArray(n);
            tShared[r] = timeThreads(n, id -> unpaddedWork(shared, id, ITERATIONS));
            check("Unpadded", shared, n, 1);

            // ---- Version B: PADDED array ----
            // Same work, but counters are 128 bytes apart, so each one lives
            // in its OWN cache line. Cores no longer invalidate each other.
            AtomicLongArray padded = new AtomicLongArray(n * PAD);
            tPadded[r] = timeThreads(n, id -> paddedWork(padded, id, ITERATIONS));
            check("Padded", padded, n, PAD);

            // ---- Version C: LOCAL variable, write once at the end ----
            AtomicLongArray local = new AtomicLongArray(n);
            tLocal[r] = timeThreads(n, id -> localWork(local, id, ITERATIONS));
            check("Local", local, n, 1);

            log(String.format("Run %d: unpadded %9.2f ms | padded %9.2f ms | local %8.2f ms",
                    r + 1, tShared[r], tPadded[r], tLocal[r]));
        }

        double shared = median(tShared);
        double padded = median(tPadded);
        double local  = median(tLocal);

        log("-".repeat(62));
        log(String.format("%-28s | %16s | %10s", "Version", "Wall Time (ms)", "Ratio"));
        log("-".repeat(62));
        log(String.format("%-28s | %16.2f | %9.2fx", "Unpadded Shared (8 B apart)", shared, 1.0));
        log(String.format("%-28s | %16.2f | %9.2fx", "Padded (128 B apart)", padded, shared / padded));
        log(String.format("%-28s | %16.2f | %9.2fx", "Local var, 1 write at end", local, shared / Math.max(local, 0.001)));
        log("-".repeat(62));
        log(String.format("Execution ratio Unpadded / Padded : %.2fx", shared / padded));
        log(String.format("Execution ratio Unpadded / Local  : %.2fx", shared / Math.max(local, 0.001)));
        log("=".repeat(62));

        try (FileWriter fw = new FileWriter(OUTPUT_FILE, StandardCharsets.UTF_8)) {
            fw.write(output.toString());
        } catch (IOException e) {
            System.out.println("Could not write " + OUTPUT_FILE + ": " + e.getMessage());
        }
        System.out.println("(Output also saved to " + OUTPUT_FILE + ")");
    }
}