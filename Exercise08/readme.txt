Exercise 8: Compile and Run

The Makefile is at the root of the repository (not inside this folder),
since it needs to build all 6 programs across Exercise01 to Exercise06 in
one place.

Targets:
- make all    compiles all 6 programs (sum_bcast, sum_scatter, sum_gather,
              sum_reduce, sum_allreduce, sum_scan) using mpicc.
- make run    builds everything (if needed) then runs all 6 programs with
              mpirun -np 4, one after another.
- make clean  removes the 6 compiled binaries.

Usage (from the repository root, inside WSL):
make all
make run
make clean

Verified: make all builds all 6 programs with no errors, and make run
executes all 6 with 4 processes, each printing Correct? = YES.

All source files (sum_bcast.c, sum_scatter.c, sum_gather.c, sum_reduce.c,
sum_allreduce.c, sum_scan.c), their exercise notes (exerciseNN.txt), output
screenshots, and the Makefile have been pushed to the GitHub repository.
