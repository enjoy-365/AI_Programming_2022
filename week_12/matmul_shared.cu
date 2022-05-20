// written by Jongsoo Kim
// Last modification: 2014-06-22
// compile options nvcc -Xptxas -v -arch sm_20 matmul_shared.cu 
// Intel(R) Xeon(R) CPU E5-2660 @ 2.20GHz and K40
// N= 1024, BLOCK_SIZE=32, 8.064341e+01 Gflops
// N= 2048, BLOCK_SIZE=32, 1.531437e+02 Gflops
// N= 4096, BLOCK_SIZE=32, 2.135609e+02 Gflops
// N= 8192, BLOCK_SIZE=32, 2.849405e+02 Gflops
// N=16384, BLOCK_SIZE=32, 3.266475e+02 Gflops

// Intel(R) Xeon(R) CPU           E5640  @ 2.67GHz
// N= 1024, BLOCK_SIZE=32, 1.223431e+02 Gflops
// N= 2048, BLOCK_SIZE=32, 1.546619e+02 Gflops
// N= 4096, BLOCK_SIZE=32, 1.656008e+02 Gflops 
// N= 8192, BLOCK_SIZE=32, 1.727456e+02 Gflops 

#include <stdio.h>  
#include <stdlib.h>  // for malloc
#include <time.h>    // for the clock() function

// size of a square matrix
const int N = 1024;

// number of threads in a block
const int BLOCK_SIZE = 32; 

__global__ void MatMul(const float * A, const float * B, float * C) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Cvalue = 0.0f;

    for (int tile=0; tile<N/BLOCK_SIZE; ++tile) {

        // loading of A and B matrices into shared memory
        As[ty][tx] = A[row * N + (tile*BLOCK_SIZE+tx)]; 
        Bs[ty][tx] = B[(tile*BLOCK_SIZE+ty)*N + col]; 
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
        Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row*N+col] = Cvalue;
}

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

    // Imvoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    MatMul<<<dimGrid,dimBlock>>>(dA, dB, dC);

    // copy C from device  to host 
    cudaMemcpy (C, dC, size, cudaMemcpyDeviceToHost );

//    for (int row=0; row<N; ++row)  
//    for (int col=0; col<N; ++col) { 
//        printf("%f\n", C[row*N+col]);
//   }

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
