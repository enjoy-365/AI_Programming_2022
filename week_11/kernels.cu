//ver 1.10.2 2021.5.17

// #include <curand_kernel.h>
// #include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define m_PI 3.14159265358979323846




// solve linear system
// A will be modified
__global__ void linear_solve_one_column(int cur_icol, int n, double *A, double *b){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;
  
  if (row >= n) return;
  if (col >= n + 1) return;

  // if (col == n) b[row] = 1;      // (n+1)-th column -> b
  // else A[row*n + col] = 1;
  
  if (row == cur_icol) return;
  // if (col == cur_icol) return;   // Synchronize issue
  if (col <= cur_icol) return;   // Synchronize issue
  if (col == n) b[row] = b[row] - b[cur_icol]*A[row*n + cur_icol]/A[cur_icol*n + cur_icol];      // (n+1)-th column -> b
  else A[row*n + col] = A[row*n + col] - A[cur_icol*n + col]*A[row*n + cur_icol]/A[cur_icol*n + cur_icol];

  // x[row*(n + 3) + col] = 1;
  // if (cur_icol == (n - 1) && col == n && row == (n - 2)) {
  //   x[0] = b[row];
  //   x[1] = b[cur_icol];
  //   x[2] = A[row*n + cur_icol];
  //   x[3] = A[cur_icol*n + cur_icol];
  // }
}

// solve linear system
// A will be modified
__global__ void linear_solve_clear_one_column(double* x, int n_col, int n, double *A, double *b){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;
  
  if (row >= n) return;
  if (col >= n) return;

  if (col != row) A[row*n + col] = 0;
  else x[row] = b[row]/A[row*n + col];
}


// solve linear system
__global__ void linear_solve_Mat_one_column(int cur_icol, int n, int n_col, double *A, double *B){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;
  
  if (row >= n) return;
  if (col >= n + n_col) return;
  
  if (row == cur_icol) return;
  if (col <= cur_icol) return;   // Synchronize issue
  if (col >= n) B[row*n_col + col - n] = B[row*n_col + col - n] - B[cur_icol*n_col + col - n]*A[row*n + cur_icol]/A[cur_icol*n + cur_icol];      // (n+1), (n+2), ..., (n+n_col)-th column -> B
  else A[row*n + col] = A[row*n + col] - A[cur_icol*n + col]*A[row*n + cur_icol]/A[cur_icol*n + cur_icol];
}

// solve linear system
__global__ void linear_solve_Mat_clear_one_column(double* X, int n, int n_col, double *A, double *B){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;
  
  if (row >= n) return;
  if (col >= n + n_col) return;

  if (col < n) {
      if (col != row) A[row*n + col] = 0;
  }
  else
      X[row*n_col + col - n] = B[row*n_col + col - n]/A[row*n + row];

}



// cuda cover test
__global__ void cuda_cover_test(float* x, int n){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  x[row*n + col] = 1;
}


__global__ void matmul(float *C, int n, const float *A, const float *B){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  // printf("row[%d], col[%d]\n", row, col);

  if(row < n && col < n){
    float val = 0.0;
    for(int i=0; i<n; ++i){
      val += A[row*n + i]*B[n*i + col];
    }
    C[row*n + col] = val;
  }
}

// calculate square of distance
__global__ void distCal(float *C, int n1, int n2, int dim, const float *A, const float *B){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  float diff;

  if(row < n1 && col < n2){
    float val = 0.0;
    for(int i=0; i<dim; ++i){
      diff = (A[dim*row + i] - B[dim*col + i]);
      val += diff*diff;
    }
    C[row*n2 + col] = val;
  }
}

__global__ void kernelvals(float *C, int n1, int n2, int dim, const float *A, const float *B, const float h_bw){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  float diff;

  if(row < n1 && col < n2){
  //if(row == 0  && col == 0){
    float val = 0.0;
    for(int i=0; i<dim; ++i){
      diff = (A[dim*row + i] - B[dim*col + i]);
      //diff = -3;
      val += diff*diff;
    }
    // C[row*n2 + col] = exp(-.5*val/pow(h_bw, 2))/pow(sqrt(2*m_PI), dim)/pow(h_bw, dim);
    C[row*n2 + col] = exp(-.5*val/pow(h_bw, 2) - (dim/2.)*log(2.*m_PI) - dim*log(h_bw));
  }
}

__global__ void logGaussian(float *logPs, int datanum, int dim, const float *data, const float *mean, const float *invCov, const float detCov){

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx_datum = bx*blockDim.x + tx;

  if(idx_datum < datanum) {
    float val = 0.0;
    for(int i=0; i<dim; ++i) {
      for(int j=0; j<dim; ++j) {
	val += (data[dim*idx_datum + j] - mean[j])*invCov[dim*j + i]*(data[dim*idx_datum + i] - mean[i]);
      }
    }
    logPs[idx_datum] = -.5*val - dim/2.*log(2.*m_PI) - 0.5*log(detCov);
  }
}

