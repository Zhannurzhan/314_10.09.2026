/*
 * collatz_phase4.c  --  Phase 4: Micro-architectural experiments
 *
 * Compile (PowerShell, MSYS2/WinLibs gcc):
 *     gcc -O2 -fopenmp collatz_phase4.c -o collatz_phase4.exe
 * Run:
 *     .\collatz_phase4.exe <N> [threads]
 * Example (Student ID ending 0984 -> N = 10,984,000):
 *     .\collatz_phase4.exe 10984000 8
 *
 * Experiment A: false sharing (naive array vs padded struct vs reduction)
 * Experiment B: loop scheduling (static / dynamic / guided)
 *
 * Every configuration is run 3 times: Run 1 = cold (discarded),
 * Avg = (Run 2 + Run 3) / 2, matching the worksheet's timing rule.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MAX_THREADS 256
#define MOD         1000000007ULL
#define THRESH      100            /* count numbers needing > 100 steps */

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

/* ---------- Experiment A data structures ---------- */

/* Naive counters: adjacent ints share one 64-byte cache line.
   volatile forces a real memory write on every increment, so the
   compiler cannot keep the counter in a register (which would hide
   the false sharing). */
static volatile int hit_count[MAX_THREADS] __attribute__((aligned(64)));

/* Padded counters: one counter per 64-byte cache line. */
struct __attribute__((aligned(64))) Padded { int count; char pad[60]; };
static struct Padded padded[MAX_THREADS];

static uint64_t run_naive(uint64_t N, int nt) {
    for (int t = 0; t < MAX_THREADS; t++) hit_count[t] = 0;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > THRESH) hit_count[omp_get_thread_num()]++;
    }
    uint64_t total = 0;
    for (int t = 0; t < nt; t++) total += hit_count[t];
    return total;
}

static uint64_t run_padded(uint64_t N, int nt) {
    for (int t = 0; t < MAX_THREADS; t++) padded[t].count = 0;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > THRESH) padded[omp_get_thread_num()].count++;
    }
    uint64_t total = 0;
    for (int t = 0; t < nt; t++) total += padded[t].count;
    return total;
}

static uint64_t run_reduction(uint64_t N, int nt) {
    uint64_t total_hits = 0;
    #pragma omp parallel for schedule(static) num_threads(nt) reduction(+:total_hits)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > THRESH) total_hits++;
    }
    return total_hits;
}

typedef uint64_t (*variant_fn)(uint64_t, int);

static void bench_variant(const char *name, variant_fn f, uint64_t N, int nt) {
    double t[3];
    uint64_t r = 0;
    for (int run = 0; run < 3; run++) {
        double s = omp_get_wtime();
        r = f(N, nt);
        t[run] = omp_get_wtime() - s;
    }
    double avg = (t[1] + t[2]) / 2.0;
    printf("%-18s threads=%-3d run1=%.4f run2=%.4f run3=%.4f avg=%.4f s | "
           "hits=%llu | throughput=%.3e iter/s\n",
           name, nt, t[0], t[1], t[2], avg,
           (unsigned long long)r, (double)N / avg);
}

/* ---------- Experiment B ---------- */

static double tbusy[MAX_THREADS];

/* Schedule is chosen at runtime via omp_set_schedule(). */
static void sched_run(uint64_t N, int nt, uint32_t *max_out,
                      uint64_t *checksum_out, uint64_t *hits_out) {
    uint32_t mx = 0;
    uint64_t sum = 0, hits = 0;
    for (int t = 0; t < MAX_THREADS; t++) tbusy[t] = 0.0;

    #pragma omp parallel num_threads(nt) reduction(max:mx) reduction(+:sum,hits)
    {
        double t0 = omp_get_wtime();
        #pragma omp for schedule(runtime) nowait
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t s = collatz_steps(i);
            if (s > mx) mx = s;
            sum += s;
            if (s > THRESH) hits++;
        }
        tbusy[omp_get_thread_num()] = omp_get_wtime() - t0;
    }
    *max_out = mx;
    *checksum_out = sum % MOD;
    *hits_out = hits;
}

static void bench_sched(const char *name, omp_sched_t kind, int chunk,
                        uint64_t N, int nt) {
    omp_set_schedule(kind, chunk);
    double t[3];
    uint32_t mx = 0;
    uint64_t cs = 0, hits = 0;
    for (int run = 0; run < 3; run++) {
        double s = omp_get_wtime();
        sched_run(N, nt, &mx, &cs, &hits);
        t[run] = omp_get_wtime() - s;
    }
    double avg = (t[1] + t[2]) / 2.0;

    /* per-thread busy time from the last run -> load imbalance */
    double tmin = tbusy[0], tmax = tbusy[0];
    for (int i = 1; i < nt; i++) {
        if (tbusy[i] < tmin) tmin = tbusy[i];
        if (tbusy[i] > tmax) tmax = tbusy[i];
    }
    printf("%-22s run1=%.4f run2=%.4f run3=%.4f avg=%.4f s | "
           "thread busy min=%.4f max=%.4f (max/min=%.2f) | "
           "max_steps=%u checksum=%llu hits=%llu\n",
           name, t[0], t[1], t[2], avg, tmin, tmax,
           tmin > 0 ? tmax / tmin : 0.0,
           mx, (unsigned long long)cs, (unsigned long long)hits);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N> [threads]\n", argv[0]);
        return 1;
    }
    uint64_t N = strtoull(argv[1], NULL, 10);
    int nt = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > MAX_THREADS) nt = MAX_THREADS;

    printf("N = %llu, threads = %d, omp_get_max_threads = %d\n\n",
           (unsigned long long)N, nt, omp_get_max_threads());

    printf("=== Experiment A: False sharing ===\n");
    bench_variant("V1 naive hits[tid]", run_naive,     N, nt);
    bench_variant("V2 padded struct",   run_padded,    N, nt);
    bench_variant("V2 reduction",       run_reduction, N, nt);

    printf("\n=== Experiment B: Loop scheduling (%d threads) ===\n", nt);
    bench_sched("static (default)",    omp_sched_static,  0,     N, nt);
    bench_sched("static, 1000",        omp_sched_static,  1000,  N, nt);
    bench_sched("dynamic, 100",        omp_sched_dynamic, 100,   N, nt);
    bench_sched("dynamic, 10000",      omp_sched_dynamic, 10000, N, nt);
    bench_sched("guided",              omp_sched_guided,  0,     N, nt);
    return 0;
}
