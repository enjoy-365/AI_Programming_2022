// written by Jongsoo Kim
// Last modification: 2014-06-04 
// compile options nvcc -arch sm_20 exec_conf_2D.cu

#include <stdio.h>

__global__ void exec_conf(void) {

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    printf("ix = %d, iy = %d, threadIdx = (%d,%d,%d), blockIdx = (%d,%d,%d), blockDim = (%d,%d,%d), gridDim = (%d,%d,%d)\n",
            ix, iy,
            threadIdx.x,threadIdx.y,threadIdx.z,
            blockIdx.x,blockIdx.y,blockIdx.z,
            blockDim.x,blockDim.y,blockDim.z,
            gridDim.x,gridDim.y,gridDim.z);

}

int main (void) {

    dim3 blocks(2,2,1);
    dim3 threads(4,2,1);
    exec_conf<<<blocks,threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
