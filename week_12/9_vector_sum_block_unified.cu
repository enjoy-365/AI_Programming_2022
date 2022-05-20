// written by Jongsoo Kim
// Last modification: 2014-06-21
// compile options nvcc -arch sm_52 vector_sum_block_unified.cu

#include <stdio.h>

const int N = 128;

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    c[tid] = a[tid] + b[tid];
}

int main (void) {

    int *a, *b, *c;
    cudaMallocManaged (&a, N); 
    cudaMallocManaged (&b, N); 
    cudaMallocManaged (&c, N); 

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    add<<<N,1>>>(a, b, c);
    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    cudaFree (a); cudaFree (b); cudaFree (c);

    return 0;
}
