CC = icc
CFLAGS = -std=c99 -Wall -O2 -g -fast -unroll4 -simd -qopenmp
LDFLAGS = -lm

jacobi: jacobi.c
	$(CC) $(CFLAGS) -o jacobi jacobi.c $(LDFLAGS)
