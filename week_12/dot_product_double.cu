// written by Jongsoo Kim
// last modification: 2011-06-28
// compile options nvcc -arch sm_13 dot_product_double.cu

#include <stdio.h>
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 256*2048;
const int threadsPerBlock = 256;
const int blocksPerGrid = N/256; 
//const int blocksPerGrid = 16; 
//const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock; 

__global__ void dot_product(double *a, double *b, double *c) { 

    __shared__ double cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
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
    
    if (cacheIndex ==0) c[blockIdx.x] = cache[0];
}

int main (void) { 

    double *a, *b, c, *partial_c;
    double *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = (double*) malloc ( N*sizeof(double) );
    b = (double*) malloc ( N*sizeof(double) );
    partial_c = (double*) malloc ( blocksPerGrid*sizeof(double) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N*sizeof(double) );
    cudaMalloc( (void**)&dev_b, N*sizeof(double) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(double) );

    for (int i=0; i<N; i++) {
       a[i] = (double) i;
       b[i] = (double) i;
    }

    cudaMemcpy ( dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N*sizeof(double), cudaMemcpyHostToDevice );

    dot_product<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

    cudaMemcpy ( partial_c, dev_partial_c, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost );

    c = 0.0;
    for (int i=0; i<blocksPerGrid; i++)
        c += partial_c[i];

    printf("Dot prodoct of a and b = %f\n", c);
    printf("sum_squares of (N-1) = %f\n", sum_squares((double)(N-1)) );

    // free memory on the CUP side
    free(a); free(b); free(partial_c);

    // free memory on the GPU side
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_partial_c);

    return 0;
}
