CC = mpicc
CFLAGS = -O2
NP = 4

BCAST     = Exercise01/sum_bcast
SCATTER   = Exercise02/sum_scatter
GATHER    = Exercise03/sum_gather
REDUCE    = Exercise04/sum_reduce
ALLREDUCE = Exercise05/sum_allreduce
SCAN      = Exercise06/sum_scan

TARGETS = $(BCAST) $(SCATTER) $(GATHER) $(REDUCE) $(ALLREDUCE) $(SCAN)

.PHONY: all run clean

all: $(TARGETS)

$(BCAST): Exercise01/sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

$(SCATTER): Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

$(GATHER): Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

$(REDUCE): Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

$(ALLREDUCE): Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

$(SCAN): Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	mpirun -np $(NP) ./$(BCAST)
	mpirun -np $(NP) ./$(SCATTER)
	mpirun -np $(NP) ./$(GATHER)
	mpirun -np $(NP) ./$(REDUCE)
	mpirun -np $(NP) ./$(ALLREDUCE)
	mpirun -np $(NP) ./$(SCAN)

clean:
	rm -f $(TARGETS)
