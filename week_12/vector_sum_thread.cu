// written by Jongsoo Kim
// Last modification: 2014-06-04
// compile options nvcc -arch sm_20 vector_sum_block.cu

#include <stdio.h>

const int N = 128;

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x;    // handle the data at this index
    c[tid] = a[tid] + b[tid];
}

int main (void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i; b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy ( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );
    
    add<<<1,N>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy ( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );

    // display the results
    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    // free the memory allocated on the CPU
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);

    return 0;
}
