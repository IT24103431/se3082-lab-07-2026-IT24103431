
![lab](/resources/pclogo-2-2.png)

## <div align="center">Lab 07</div>


## Objectives:

* Understand MPI collective communication operations
* Use MPI_Bcast, MPI_Scatter, MPI_Gather, MPI_Reduce, MPI_Allreduce, and MPI_Scan
* Progressively refactor a parallel array sum program using different collectives

## The Problem

All exercises in this lab solve the **same problem**: sum an integer array of **1,000,000 elements** (values 1 to 1,000,000) in parallel using MPI. The expected answer is **500,000,500,000**.

You will start with a complete working program using `MPI_Bcast`, then progressively replace parts of it with more efficient collectives. Each exercise builds on the previous one.

**Compile:** `mpicc -o program program.c`  
**Run:** `mpirun -np 4 ./program`

> **Note:** The number of processes must evenly divide 1,000,000. Use 4, 5, 8, 10, etc.

---

## Exercise 1: Broadcast + Send/Recv (Given Program)

This complete program is your starting point. Study it carefully before proceeding.

**Strategy:** Root fills the array and broadcasts the **entire** array to all processes. Each process sums its own portion. Non-root processes send their partial sums back to root using `MPI_Send`; root receives and accumulates with `MPI_Recv`.

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Every process allocates the FULL array.
     * This is the downside of broadcast — wastes memory.
     */
    int *array = (int *)malloc(N * sizeof(int));

    /* Root fills the array with values 1 to N */
    if (rank == 0) {
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    /*
     * BROADCAST: Root sends the entire array to ALL processes.
     * Every process must call this — not just root!
     * After this call, every process has a complete copy of array[].
     */
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * Each process computes the sum of its own portion.
     * Process i handles indices [i * chunk_size .. (i+1) * chunk_size - 1]
     */
    int chunk_size = N / size;
    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;

    long long local_sum = 0;
    for (int i = start_idx; i < end_idx; i++)
        local_sum += array[i];

    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

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
        printf("\n[Bcast] Total sum   = %lld\n", total_sum);
        printf("[Bcast] Expected    = %lld\n", expected);
        printf("[Bcast] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time        = %.4f sec\n", elapsed);
    }

    free(array);
    MPI_Finalize();
    return 0;
}
```

**Things to note:**

* Every process allocates and receives the **full** 1M-element array, even though each only needs 250K elements (with 4 processes). This wastes memory.
* The Send/Recv loop on root is O(P) — root receives from each process one by one.
* All processes must call `MPI_Bcast`, not just root. Putting it inside `if (rank == 0)` would cause a deadlock.

---

## Exercise 2: Replace Broadcast with Scatter

In Exercise 1, every process received the **entire** array via `MPI_Bcast`, even though each process only works on its own chunk. `MPI_Scatter` is more efficient — it divides the array on root and sends **only the relevant chunk** to each process.

**MPI_Scatter signature:**

```c
int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
```

* `sendbuf` — the full array (only meaningful on root; can be NULL on other processes)
* `sendcount` — number of elements to send **to each process** (NOT the total array size!)
* `recvbuf` — local buffer where each process receives its chunk
* `recvcount` — number of elements each process receives (usually equals `sendcount`)
* `root` — the process that holds the full array

**What you need to do:**

1. Copy your Exercise 1 program to a new file `sum_scatter.c`.
2. Change the memory allocation so that **only root** allocates the full array. All processes should allocate a small `local_chunk` buffer of size `N / size`.
3. Replace the `MPI_Bcast` call with `MPI_Scatter` to distribute chunks.
4. Modify the local sum loop to iterate over `local_chunk[0..chunk_size)` instead of `array[start_idx..end_idx)`.
5. Keep the Send/Recv collection unchanged for now.
6. Verify that the total sum is still **500,000,500,000**.

> **Common mistake:** Setting `sendcount` to `N` instead of `N / size`. The `sendcount` is the number of elements sent to **each** process, not the total.

---

## Exercise 3: Replace Send/Recv with Gather

In Exercise 2, the partial sums are collected at root using a manual `MPI_Send`/`MPI_Recv` loop. `MPI_Gather` replaces this entire loop with a single call — it collects one value from each process and assembles them in rank order on root.

**MPI_Gather signature:**

```c
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
```

* `sendbuf` — data each process sends (e.g., `&local_sum`)
* `sendcount` — number of elements each process sends (1 in our case)
* `recvbuf` — buffer on root where gathered data is placed in rank order (only meaningful on root)
* `recvcount` — number of elements received **from each** process (1 in our case)
* `root` — the process that collects the data

**What you need to do:**

1. Copy your Exercise 2 program to a new file `sum_gather.c`.
2. On root, allocate an array `all_sums` of size `size` (one `long long` per process).
3. Replace the entire Send/Recv block with a single `MPI_Gather` call that collects `local_sum` from every process into `all_sums` on root.
4. On root, loop through `all_sums` to compute the total.
5. Verify the result.

> **Note:** Gather does **not** perform any computation — it only collects data. Root still needs to manually sum the gathered values.

---

## Exercise 4: Replace Gather with Reduce

In Exercise 3, root gathers all partial sums and then manually adds them in a loop. `MPI_Reduce` does both in one step — it combines all local values using an operation (like `MPI_SUM`) and places the result on root. Internally, MPI uses a tree-based algorithm that runs in O(log P) steps.

**MPI_Reduce signature:**

```c
int MPI_Reduce(void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op,
               int root, MPI_Comm comm);
```

* `sendbuf` — value each process contributes (e.g., `&local_sum`)
* `recvbuf` — where the result is stored (only valid on root!)
* `count` — number of elements to reduce (1 for a single sum)
* `op` — the reduction operation: `MPI_SUM`, `MPI_MAX`, `MPI_MIN`, `MPI_PROD`, etc.
* `root` — the process that receives the result

**What you need to do:**

1. Copy your Exercise 3 program to a new file `sum_reduce.c`.
2. Remove the `all_sums` array and the manual summation loop on root.
3. Replace the `MPI_Gather` call with `MPI_Reduce` using `MPI_SUM` as the operation, with root = 0.
4. On root, the `total_sum` variable will already contain the correct answer — no further computation needed.
5. Verify the result.

> **Important:** After `MPI_Reduce`, the result in `recvbuf` is **only valid on root**. If a non-root process tries to use `total_sum`, it will contain garbage. If all processes need the result, see Exercise 5.

---

## Exercise 5: Replace Reduce with Allreduce

`MPI_Reduce` delivers the result to root only. If all processes need the total (e.g., to compute their percentage contribution), you would need a follow-up `MPI_Bcast`. `MPI_Allreduce` does both in a single, more efficient call — every process gets the result.

**MPI_Allreduce signature:**

```c
int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op,
                  MPI_Comm comm);
```

* Same as `MPI_Reduce`, but with **no root parameter** — the result goes to all processes.

**What you need to do:**

1. Copy your Exercise 4 program to a new file `sum_allreduce.c`.
2. Replace `MPI_Reduce` with `MPI_Allreduce` (remove the root parameter).
3. After the call, **every** process now has the correct `total_sum`.
4. Add the following on **every** process (not just root): each process should print its `local_sum`, the `total_sum`, and what **percentage** of the total its chunk contributed.
5. Keep the verification print on root.
6. Verify the result.

> **Best practice:** Prefer `MPI_Allreduce` over `MPI_Reduce` + `MPI_Bcast` when all processes need the result. It is typically implemented more efficiently (butterfly/recursive-doubling algorithm).

---

## Exercise 6: Prefix Sums with Scan

`MPI_Scan` computes a **prefix reduction** — each process receives the cumulative result of the operation applied to all values from rank 0 up to and including its own rank. Unlike Allreduce (where every process gets the same answer), each process gets a **different** result.

**MPI_Scan signature:**

```c
int MPI_Scan(void *sendbuf, void *recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm);
```

* Same parameters as `MPI_Allreduce`. No root parameter.
* After the call with `MPI_SUM`:
  * Rank 0 receives: `local_sum_0`
  * Rank 1 receives: `local_sum_0 + local_sum_1`
  * Rank 2 receives: `local_sum_0 + local_sum_1 + local_sum_2`
  * Last rank receives: total sum of entire array

**What you need to do:**

1. Copy your Exercise 5 program to a new file `sum_scan.c`.
2. Replace `MPI_Allreduce` with `MPI_Scan`. Store the result in a variable called `prefix_sum`.
3. Each process should print:
   * Its `local_sum`
   * Its `prefix_sum` (the cumulative sum of all chunks from rank 0 through this rank)
   * The value `sum_before_me = prefix_sum - local_sum` (the total of all chunks before this process)
4. Verify: The last rank's `prefix_sum` should equal **500,000,500,000** (the global total).
5. **Verification bonus:** For our array of values 1 to N, the sum of elements 1 through K is `K*(K+1)/2`. Each process can verify its `prefix_sum` against this formula using `K = (rank + 1) * chunk_size`.

**Practical use case to think about:** Each process's `sum_before_me` acts as a **global offset**. If you needed to compute running totals within each chunk that are globally correct (not just local), you would add `sum_before_me` to each local prefix sum. This is how Scan enables global index assignment, cumulative distributions, and load-balanced work splitting — without any extra communication.

---

## Exercise 7: Summary Comparison

1. Create a table comparing all 6 programs you wrote. For each program, note:
   * Which collectives were used
   * Whether every process allocates the full array or just a chunk
   * Whether root needs a manual summation loop
   * Where the final result is available (root only, or all processes, or different per rank)

2. Run all 6 programs with 2, 4, and 8 processes. Record the execution times and note which approach is fastest. Briefly explain why.

3. **Thinking question:** In what situation would you choose `MPI_Scan` over `MPI_Allreduce`? Give a concrete example.

---

## Exercise 8: Compile and Run

1. Create a `Makefile` that compiles all 6 programs and has a `run` target that executes them all with 4 processes.

2. Push all your source files and the Makefile to your GitHub repository.





