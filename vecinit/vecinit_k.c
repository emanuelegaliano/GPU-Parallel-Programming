// Programma intermedio (transizione da C a OpenCL C)

#include <stdio.h>
#include <stdlib.h>

/*
  Futuro kernel device: 
    Trasformiamo il corpo del ciclo for contenuto nella funzione init() in una funzione a sé stante che
    prende in input un vettore e l'indice dell'elemento da modificare (per quella iterazione).
*/
void init_k(int i, int *vec)
{
	vec[i] = i;
}

void init(int *vec, int nels)
{
	for (int i = 0; i < nels; ++i)
		init_k(i, vec);
}


void verify(const int *vec, int nels) {
	for (int i = 0; i < nels; ++i) {
		int expected = i;
		int computed = vec[i];
		if (expected != computed) {
			fprintf(stderr, "%d != %d @ %d\n",
				expected, computed, i);
			exit(7);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stderr, "%s nels\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}

	int *vec = malloc(sizeof(int)*nels);
	if (!vec) {
		fprintf(stderr, "allocazione fallita\n");
		exit(3);
	}

	init(vec, nels);

	verify(vec, nels);

	return 0;
}
