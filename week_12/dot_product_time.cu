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
    a = (float*) malloc ( N*sizeof(float) );
    b = (float*) malloc ( N*sizeof(float) );
    partial_c = (float*) malloc ( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N*sizeof(float) );
    cudaMalloc( (void**)&dev_b, N*sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float) );

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    cudaEventRecord ( start, 0 );

    cudaMemcpy ( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice );

    dot_product<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

    cudaMemcpy ( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime ( &elapsedTime, start, stop );

    printf( "Elapsed time: %3.1f ms\n", elapsedTime );

    c = 0.0f;
    for (int i=0; i<blocksPerGrid; i++)
        c += partial_c[i];

    // free memory on the CUP side
    free(a); free(b); free(partial_c);

    // free memory on the GPU side
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_partial_c);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );


}
