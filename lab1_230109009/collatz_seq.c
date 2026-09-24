#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#define N 19009000ULL
#define MOD 1000000007ULL

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;

    while (n > 1) {
        if ((n & 1) == 0)
            n >>= 1;
        else
            n = 3 * n + 1;

        steps++;
    }

    return steps;
}

int main(void) {
    uint64_t checksum = 0;
    uint32_t max_steps = 0;

    double start = omp_get_wtime();

    for (uint64_t i = 1; i <= N; i++) {
        uint32_t steps = collatz_steps(i);

        if (steps > max_steps)
            max_steps = steps;

        checksum = (checksum + steps) % MOD;
    }

    double end = omp_get_wtime();

    printf("N = %llu\n", (unsigned long long)N);
    printf("Max stopping time = %u\n", max_steps);
    printf("Checksum = %llu\n", (unsigned long long)checksum);
    printf("Execution time = %.9f seconds\n", end - start);

    return 0;
}