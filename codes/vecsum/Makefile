CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -lOpenCL

all: vecsum vecsum_vect

vecsum: vecsum.c
	$(CC) $(CFLAGS) -o vecsum vecsum.c $(LDFLAGS)

vecsum_vect: vecsum_vect.c
	$(CC) $(CFLAGS) -o vecsum_vect vecsum_vect.c $(LDFLAGS)

clean:
	rm -f vecsum vecsum_vect

.PHONY: clean