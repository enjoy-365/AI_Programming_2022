// written by Jongsoo Kim
// Last modification: 2014-06-22
// compile options nvcc -arch sm_20 -l cublas matmul_cublas.cu 
// Intel(R) Xeon(R) CPU E5-2660 @ 2.20GHz and K40
// N= 1024, BLOCK_SIZE=32, 7.482187e+01 Gflops
// N= 2048, BLOCK_SIZE=32, 1.030203e+02 Gflops
// N= 4096, BLOCK_SIZE=32, 1.208879e+02 Gflops
// N= 8192, BLOCK_SIZE=32, 1.278362e+02 Gflops
// N=16384, BLOCK_SIZE=32, 1.292009e+02 Gflops

// compile options nvcc -O3 -arch sm_20 matmul_global.cu         
// Intel(R) Xeon(R) CPU E5640  @ 2.67GHz and Tesla S2050 
// N= 1024, BLOCK_SIZE=32, 1.028392e+02 Gflops
// N= 2048, BLOCK_SIZE=32, 1.207798e+02 Gflops
// N= 4096, BLOCK_SIZE=32, 1.349969e+02 Gflops
// N= 8192, BLOCK_SIZE=32, 1.315237e+02 Gflops

// compile options nvcc -O3 -arch sm_20 matmul_cublas.cu
// Intel(R) Xeon(R) CPU E5640  @ 2.67GHz and Tesla S2050         
// N= 1024, 1.019030e+01 Gflops
// N= 2048, 7.185082e+01 Gflops
// N= 4096, 2.888868e+02 Gflops
// N= 8192, 5.031677e+02 Gflops

#include <stdio.h>  
#include <stdlib.h>  // for malloc
#include "cublas_v2.h"

// size of a square matrix
const int N = 8192;

int main (void) {

    float * A, * B, *C;    // arrays for host
    float * dA, * dB, *dC; // arrays for device 

    size_t size = N * N * sizeof(float);

    A = (float *) malloc(size);
    B = (float *) malloc(size);
    C = (float *) malloc(size);

    cudaMalloc( (void**)&dA,size);
    cudaMalloc( (void**)&dB,size);
    cudaMalloc( (void**)&dC,size);

    // initialization of A and B matrices
    for (unsigned row=0; row<N; ++row)  
    for (unsigned col=0; col<N; ++col) { 
        A[row * N + col] = row + col + 1.0f; 
        B[row * N + col] = row + col + 1.0f; 
    }

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start,0 );

    // copy A and B from host to device
    cudaMemcpy (dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy (dB, B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle=0;
    cublasCreate(&handle);
    int lda=N,ldb=N,ldc=N;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    cublasDestroy(handle);

    // copy C from device  to host 
    cudaMemcpy (C, dC, size, cudaMemcpyDeviceToHost );

//    for (int row=0; row<N; ++row)  
//    for (int col=0; col<N; ++col) { 
//        printf("%f\n", C[row*N+col]);
//    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float etime;
    cudaEventElapsedTime( &etime, start, stop );
    etime = etime/1000.0;

    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = 2.0f*N*N*N;
    printf("number of multiplications and additions  = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    free(A); free(B); free(C);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}
