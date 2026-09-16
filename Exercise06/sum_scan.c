#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    /*
     * Only root allocates the FULL array.
     * Every process (including root) allocates a small local_chunk buffer.
     */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /* SCATTER: distribute one chunk of the array to each process */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process sums its own local chunk */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    /*
     * SCAN: prefix reduction. Each process receives the cumulative sum of
     * local_sum from rank 0 up to and including its own rank. Unlike
     * Allreduce, every process gets a DIFFERENT result. No root parameter.
     */
    long long prefix_sum = 0;
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    long long sum_before_me = prefix_sum - local_sum;

    printf("Rank %d: local_sum = %lld, prefix_sum = %lld, sum_before_me = %lld\n",
           rank, local_sum, prefix_sum, sum_before_me);

    /* Verification bonus: sum of 1..K is K*(K+1)/2, where K is the number
     * of elements covered by ranks 0 through this rank. */
    long long K = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = K * (K + 1) / 2;
    printf("  Rank %d: K = %lld, formula K*(K+1)/2 = %lld, matches prefix_sum? %s\n",
           rank, K, expected_prefix, prefix_sum == expected_prefix ? "YES" : "NO");

    double elapsed = MPI_Wtime() - start;

    /* The last rank's prefix_sum is the global total. */
    if (rank == size - 1) {
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scan] Final prefix_sum (last rank) = %lld\n", prefix_sum);
        printf("[Scan] Expected                     = %lld\n", expected);
        printf("[Scan] Correct?                     = %s\n", prefix_sum == expected ? "YES" : "NO");
        printf("[Scan] Time                          = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0)
        free(array);

    MPI_Finalize();
    return 0;
}
