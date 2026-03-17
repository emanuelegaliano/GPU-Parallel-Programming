#include <stdio.h>
#include <stdlib.h>

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

    /*
    TODO List:
    1. Set up device on openCL
    2. Allocate memory on device
    3. Get the kernel code and compile it
    4. Set kernel data available on device
    */


}