// last modification: 2011-06-28
// compile options nvcc dot_product.cu

#include <stdio.h>
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 33*1024*2048;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 2; 
//const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock; 

__global__ void dot_product(float *a, float *b, float *c) { 

    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // becuase of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex ==0)
        c[blockIdx.x] = cache[0];

}

int main (void) {

    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    // allocate memory on the CPU side
    cudaHostAlloc( (void**)&a, N*sizeof(float), 
                    cudaHostAllocWriteCombined | cudaHostAllocMapped );
    cudaHostAlloc( (void**)&b, N*sizeof(float), 
                    cudaHostAllocWriteCombined | cudaHostAllocMapped );
    cudaHostAlloc( (void**)&partial_c, N*sizeof(float), 
                    cudaHostAllocWriteCombined | cudaHostAllocMapped );

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    cudaHostGetDevicePointer ( &dev_a, a, 0 );
    cudaHostGetDevicePointer ( &dev_b, a, 0 );
    cudaHostGetDevicePointer ( &dev_partial_c, a, 0 );

    cudaEventRecord ( start, 0 );

    cudaMemcpy ( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice );

    dot_product<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

    cudaThreadSynchronize();

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime ( &elapsedTime, start, stop );

    c = 0.0f;
    for (int i=0; i<blocksPerGrid; i++)
        c += partial_c[i];

    // free memory on the CUP side
    cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(partial_c);

    // free memory on the GPU side
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_partial_c);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf( "Time using zero-copy: %3.1f ms\n", elapsedTime );
}
