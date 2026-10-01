/*
 * collatz_par.c  --  Phase 3: OpenMP parallel Collatz stopping time
 *
 * Compile (PowerShell):
 *     gcc -O2 -fopenmp collatz_par.c -o collatz_par.exe
 * Run (one run per launch, same method as collatz_seq):
 *     .\collatz_par.exe <N> <threads>
 * Example:
 *     .\collatz_par.exe 19009000 4
 *
 * Output must match the sequential program for the same N:
 *     Max stopping time = 704, Checksum = 78178256 (for N = 19009000)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MOD 1000000007ULL

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <threads>\n", argv[0]);
        return 1;
    }
    uint64_t N = strtoull(argv[1], NULL, 10);
    int nt = atoi(argv[2]);
    if (nt < 1) nt = 1;
    omp_set_num_threads(nt);

    uint32_t max_steps = 0;
    uint64_t sum = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(max:max_steps) reduction(+:sum)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) max_steps = s;
        sum += s;
    }

    double elapsed = omp_get_wtime() - start;

    printf("N = %llu\n", (unsigned long long)N);
    printf("Threads = %d\n", nt);
    printf("Max stopping time = %u\n", max_steps);
    printf("Checksum = %llu\n", (unsigned long long)(sum % MOD));
    printf("Execution time = %.9f seconds\n", elapsed);
    return 0;
}
