#include <stdio.h>
#include <stdlib.h>

void init(int* vec, size_t n) {
    for(size_t i = 0; i < n; i++) {
        vec[i] = i;
    }
}

void verify(const int* vec, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if(vec[i] != i) {
            fprintf(stderr, "Error: Array not initialized correctly at index %zu.\n", i);
            exit(1);
        } else {
            printf("%d ", vec[i]);
        }
    }

    puts("");
}

int main(int argc, char** argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <number>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if(n < 1) {
        fprintf(stderr, "Please provide a positive integer.\n");
        return 2;
    }

    int* arr = (int*)malloc(n * sizeof(int));
    if(arr == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 3;
    }

    // Initialize the array
    init(arr, n);

    // verifying
    verify(arr, n);

    return 0;
}