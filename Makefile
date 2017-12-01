CC = icc
CFLAGS = -std=c99 -Wall -O3 -qopenmp
LDFLAGS = -lm

jacobi: jacobi.c
	$(CC) $(CFLAGS) -o jacobi jacobi.c $(LDFLAGS)
