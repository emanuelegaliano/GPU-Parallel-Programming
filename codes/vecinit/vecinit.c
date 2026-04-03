// Programma in C
  // 1. Alloca un vettore di interi di dimensione scelta da riga di comando;
  // 2. Inizializza il vettore impostando ogni elemento al proprio indice;
  // 3. Verifica che il vettore contenga i valori desiderati.

#include <stdio.h>
#include <stdlib.h>

// Inizializzazione vettore
void init(int* vec, int nels)
{
	// futuro kernel device
	for (int i = 0; i < nels; ++i) {
		vec[i] = i;
	}
}

// Verifica del contenuto del vettore
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
  
  // Dimensione del vettore (indicata da riga di comando
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
