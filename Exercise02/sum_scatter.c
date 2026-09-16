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

    /*
     * SCATTER: Root divides the array into equal chunks and sends
     * exactly one chunk to each process (including itself).
     * sendcount is the number of elements sent to EACH process,
     * not the total array size.
     */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process sums its own local chunk */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: summed local chunk of %d elements => local_sum = %lld\n",
           rank, chunk_size, local_sum);

    /* Collect results using point-to-point communication */
    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; r++) {
            long long recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scatter] Total sum   = %lld\n", total_sum);
        printf("[Scatter] Expected    = %lld\n", expected);
        printf("[Scatter] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Scatter] Time        = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0)
        free(array);

    MPI_Finalize();
    return 0;
}
