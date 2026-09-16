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

    printf("  Rank %d: summed local chunk of %d elements => local_sum = %lld\n",
           rank, chunk_size, local_sum);

    /*
     * REDUCE: combine local_sum from every process using MPI_SUM and
     * place the result in total_sum on root. This replaces both the
     * Gather call AND the manual summation loop from Exercise 3 in a
     * single step, using an O(log P) tree-based algorithm internally.
     */
    long long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* total_sum is only valid on root after MPI_Reduce */
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Reduce] Total sum   = %lld\n", total_sum);
        printf("[Reduce] Expected    = %lld\n", expected);
        printf("[Reduce] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Reduce] Time        = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0)
        free(array);

    MPI_Finalize();
    return 0;
}
