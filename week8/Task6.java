/*
 * TASK 6 - Synchronization Cost & Data Loss
 *
 * Run:  java Task6.java
 */

import java.io.FileWriter;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class Task6 {

    // ===================== CHANGE THIS =====================
    static final String STUDENT_ID = "230109009";
    // =======================================================

    static final int THREADS = 4;
    static final int INCREMENTS = 2_000_000;          // per thread
    static final int EXPECTED = THREADS * INCREMENTS; // 8,000,000
    static final int TRIALS = 5;                      // each version runs 5 times
    static final String OUTPUT_FILE = "task6_output.txt";

    // ---- The three counters ----
    // 1) UNSAFE: no lock, no atomic, no memory fence.
    //    "Opaque" access only forces every read and write to really happen in
    //    memory (so the JIT cannot hide the race by keeping the counter in a
    //    CPU register). It gives NO protection: counter++ is still three
    //    separate steps READ -> ADD 1 -> WRITE, and threads overwrite each other.
    static int unsafeCounter = 0;
    static final VarHandle UNSAFE;
    static {
        try {
            UNSAFE = MethodHandles.lookup().findStaticVarHandle(Task6.class, "unsafeCounter", int.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    // 2) ATOMIC: a hardware atomic instruction (LOCK XADD on x86).
    static final AtomicInteger atomicCounter = new AtomicInteger(0);

    // 3) MUTEX: a classic lock around the increment.
    static int lockedCounter = 0;
    static final ReentrantLock lock = new ReentrantLock();

    static final StringBuilder output = new StringBuilder();

    /** Print to the console AND remember the line for task6_output.txt. */
    static void log(String text) {
        System.out.println(text);
        output.append(text).append("\n");
    }

    /** Start all threads at the same moment, wait for them, return time in ms. */
    static double runThreads(Runnable work) throws InterruptedException {
        CountDownLatch go = new CountDownLatch(1);
        Thread[] threads = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            threads[i] = new Thread(() -> {
                try { go.await(); } catch (InterruptedException e) { return; }
                work.run();
            });
            threads[i].start();
        }
        long start = System.nanoTime();
        go.countDown();                       // release all threads together
        for (Thread t : threads) t.join();
        return (System.nanoTime() - start) / 1_000_000.0;
    }

    static void unsafeWork() {
        for (int k = 0; k < INCREMENTS; k++) {
            int value = (int) UNSAFE.getOpaque();     // 1. READ
            UNSAFE.setOpaque(value + 1);              // 2. ADD 1  3. WRITE   <-- RACE!
        }
    }

    static void atomicWork() {
        for (int k = 0; k < INCREMENTS; k++) atomicCounter.incrementAndGet();
    }

    static void lockWork() {
        for (int k = 0; k < INCREMENTS; k++) {
            lock.lock();
            try { lockedCounter++; } finally { lock.unlock(); }
        }
    }

    public static void main(String[] args) throws Exception {
        log("=".repeat(70));
        log("TASK 6 - Synchronization Cost & Data Loss");
        log("=".repeat(70));
        log("Student ID         : " + STUDENT_ID);
        log("OS                 : " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        log("Java version       : " + System.getProperty("java.version"));
        log("Logical cores      : " + Runtime.getRuntime().availableProcessors());
        log("Threads            : " + THREADS);
        log(String.format("Increments/thread  : %,d", INCREMENTS));
        log(String.format("Expected final     : %,d", EXPECTED));
        log("Trials per version : " + TRIALS);
        log("=".repeat(70));

        // Warm-up (not reported): lets the JIT compile all three loops first.
        runThreads(Task6::unsafeWork);
        runThreads(Task6::atomicWork);
        runThreads(Task6::lockWork);

        double[] tUnsafe = new double[TRIALS], tAtomic = new double[TRIALS], tLock = new double[TRIALS];
        long totalLost = 0;

        // ---------------- Part 1: UNSAFE (race condition) ----------------
        log("PART 1 - UNSAFE counter (no lock, no atomic)");
        log(String.format("%-7s | %12s | %12s | %12s | %8s | %10s",
                "Trial", "Expected", "Actual", "Lost", "Lost %", "Time ms"));
        log("-".repeat(70));
        for (int t = 0; t < TRIALS; t++) {
            unsafeCounter = 0;
            tUnsafe[t] = runThreads(Task6::unsafeWork);
            int actual = unsafeCounter;
            int lost = EXPECTED - actual;
            totalLost += lost;
            log(String.format("%-7d | %,12d | %,12d | %,12d | %7.2f%% | %10.2f",
                    t + 1, EXPECTED, actual, lost, 100.0 * lost / EXPECTED, tUnsafe[t]));
        }
        double avgLostPct = 100.0 * totalLost / ((double) EXPECTED * TRIALS);
        log("-".repeat(70));
        log(String.format("Average updates lost: %.2f%%", avgLostPct));
        log("=".repeat(70));

        // ---------------- Part 2: CORRECT versions ----------------
        log("PART 2 - CORRECT counters");
        log(String.format("%-7s | %-24s | %-24s", "Trial", "AtomicInteger", "ReentrantLock (mutex)"));
        log("-".repeat(70));
        for (int t = 0; t < TRIALS; t++) {
            atomicCounter.set(0);
            tAtomic[t] = runThreads(Task6::atomicWork);
            lockedCounter = 0;
            tLock[t] = runThreads(Task6::lockWork);
            int a = atomicCounter.get(), l = lockedCounter;
            log(String.format("%-7d | %,10d %s %8.2f ms | %,10d %s %8.2f ms",
                    t + 1, a, a == EXPECTED ? "OK " : "BAD", tAtomic[t],
                    l, l == EXPECTED ? "OK " : "BAD", tLock[t]));
            if (a != EXPECTED || l != EXPECTED) throw new IllegalStateException("Correct version lost updates!");
        }
        log("-".repeat(70));
        log("Both correct versions reached exactly " + String.format("%,d", EXPECTED) + " every time (0% lost).");
        log("=".repeat(70));

        // ---------------- Part 3: time penalty ----------------
        double u = avg(tUnsafe), at = avg(tAtomic), lk = avg(tLock);
        log("PART 3 - Wall-clock time penalty of synchronization (average of trials)");
        log(String.format("%-28s | %10s | %12s | %8s", "Version", "Time ms", "Penalty ms", "Slowdown"));
        log("-".repeat(70));
        log(String.format("%-28s | %10.2f | %12s | %7.2fx", "Unsafe (wrong result)", u, "-", 1.0));
        log(String.format("%-28s | %10.2f | %+12.2f | %7.2fx", "AtomicInteger (hardware)", at, at - u, at / u));
        log(String.format("%-28s | %10.2f | %+12.2f | %7.2fx", "ReentrantLock (mutex)", lk, lk - u, lk / u));
        log("=".repeat(70));
        log(String.format("SUMMARY: unsafe lost %.2f%% of updates; AtomicInteger is correct and costs "
                + "%+.2f ms (%.2fx), the lock costs %+.2f ms (%.2fx).", avgLostPct, at - u, at / u, lk - u, lk / u));
        log("=".repeat(70));

        try (FileWriter fw = new FileWriter(OUTPUT_FILE, StandardCharsets.UTF_8)) {
            fw.write(output.toString());
        } catch (IOException e) {
            System.out.println("Could not write " + OUTPUT_FILE + ": " + e.getMessage());
        }
        System.out.println("(Output also saved to " + OUTPUT_FILE + ")");
    }

    static double avg(double[] v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.length;
    }
}