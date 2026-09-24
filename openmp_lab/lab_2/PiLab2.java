import java.io.PrintWriter;
import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.LongStream;

public class PiLab2 {

    interface Job { double run() throws Exception; }

    static PrintWriter csv;
    static volatile double sink;   // keeps warm-up results "used"

    static void emit(String line) {
        System.out.println(line);
        csv.println(line);
    }

    // Runs a job `trials` times, one CSV row per trial.
    // Locale.ROOT guarantees "." decimals, so the CSV parses on any Windows locale.
    static void bench(String variant, int p, int trials, Job job) throws Exception {
        for (int t = 1; t <= trials; t++) {
            long t0 = System.nanoTime();
            double pi = job.run();
            double ms = (System.nanoTime() - t0) / 1e6;
            emit(String.format(Locale.ROOT, "%s,%d,%d,%.12f,%.3e,%.3f",
                    variant, p, t, pi, Math.abs(pi - Math.PI), ms));
        }
    }

    // ---------- Serial baseline ----------
    static double runSerial(long n) {
        double step = 1.0 / n, sum = 0.0;
        for (long i = 0; i < n; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        return sum * step;
    }

    // ---------- Variant A: naive race (no synchronization) ----------
    static double runNaiveRace(long n, int threads) throws InterruptedException {
        final double step = 1.0 / n;
        final double[] shared = new double[1];
        Thread[] ts = new Thread[threads];
        long chunk = n / threads;
        for (int t = 0; t < threads; t++) {
            final long start = t * chunk;
            final long end = (t == threads - 1) ? n : start + chunk;
            ts[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * step;
                    shared[0] += 4.0 / (1.0 + x * x);   // unprotected shared write
                }
            });
            ts[t].start();
        }
        for (Thread t : ts) t.join();
        return shared[0] * step;
    }

    // ---------- Variant B: critical section (lock per step) ----------
    static double runCritical(long n, int threads) throws InterruptedException {
        final double step = 1.0 / n;
        final Object lock = new Object();
        final double[] shared = new double[1];
        Thread[] ts = new Thread[threads];
        long chunk = n / threads;
        for (int t = 0; t < threads; t++) {
            final long start = t * chunk;
            final long end = (t == threads - 1) ? n : start + chunk;
            ts[t] = new Thread(() -> {
                for (long i = start; i < end; i++) {
                    double x = (i + 0.5) * step;
                    double term = 4.0 / (1.0 + x * x);
                    synchronized (lock) { shared[0] += term; }
                }
            });
            ts[t].start();
        }
        for (Thread t : ts) t.join();
        return shared[0] * step;
    }

    // ---------- Variant C: parallel reduction in a pool of exactly P threads ----------
    static double runReduction(long n, ForkJoinPool pool) throws Exception {
        final double step = 1.0 / n;
        return pool.submit(() ->
            LongStream.range(0, n).parallel().mapToDouble(i -> {
                double x = (i + 0.5) * step;
                return 4.0 / (1.0 + x * x);
            }).sum() * step
        ).get();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: java PiLab2 <serial|race|critical|reduction> [N] [trials]");
            return;
        }
        String mode = args[0];
        long n = args.length > 1 ? Long.parseLong(args[1]) : 100_000_000L;
        int trials = args.length > 2 ? Integer.parseInt(args[2]) : 5;

        // JIT warm-up
        final long W = 2_000_000L;
        for (int k = 0; k < 3; k++) {
            sink = runSerial(W);
            sink = runNaiveRace(W, 2);
            sink = runCritical(W, 2);
        }

        csv = new PrintWriter("data_" + mode + ".csv");
        try {
            emit("variant,P,trial,pi,abs_error,time_ms");
            switch (mode) {
                case "serial" -> bench("serial", 1, trials, () -> runSerial(n));
                case "race" -> {
                    for (int p : new int[]{1, 2, 4, 8})
                        bench("race", p, trials, () -> runNaiveRace(n, p));
                }
                case "critical" -> {
                    bench("serial", 1, trials, () -> runSerial(n));
                    for (int p : new int[]{1, 2, 4, 8})
                        bench("critical", p, trials, () -> runCritical(n, p));
                }
                case "reduction" -> {
                    bench("serial", 1, trials, () -> runSerial(n));
                    for (int p : new int[]{1, 2, 4, 8, 12, 16}) {
                        ForkJoinPool pool = new ForkJoinPool(p);
                        try {
                            sink = runReduction(W, pool);   // per-pool warm-up
                            bench("reduction", p, trials, () -> runReduction(n, pool));
                        } finally {
                            pool.shutdown();
                        }
                    }
                }
                default -> System.err.println("unknown mode: " + mode);
            }
        } finally {
            csv.close();
        }
    }
}