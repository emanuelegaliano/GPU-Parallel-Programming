CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -lOpenCL

vecsum: vecsum.c
	$(CC) $(CFLAGS) -o vecsum vecsum.c $(LDFLAGS)

clean:
	rm -f vecsum

.PHONY: clean