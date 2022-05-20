// written by Jongsoo Kim
// last modification: 2014-06-21
// compile options nvcc -arch sm_20 -l cublas dot_product.cu

#include <stdio.h>
#include "cublas_v2.h"

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 2048;

int main (void) { 

    float *a, *b, c;
    float *dev_a, *dev_b;
    cublasHandle_t handle=0;

    // allocate memory on the CPU side
    a = (float*) malloc ( N*sizeof(float) );
    b = (float*) malloc ( N*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N*sizeof(float) );
    cudaMalloc( (void**)&dev_b, N*sizeof(float) );

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    cudaMemcpy ( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice );

    cublasCreate(&handle); //cuBlas library initialization
    cublasSdot(handle, N, dev_a, 1, dev_b, 1, &c);
    
    printf("Dot prodoct of a and b = %f\n", c);
    printf("sum_squares of (N-1) = %f\n", sum_squares((float)(N-1)) );

    cublasDestroy(handle);
    // free memory on the CUP side
    free(a); free(b);

    // free memory on the GPU side
    cudaFree(dev_a); cudaFree(dev_b);

    return 0;
}
