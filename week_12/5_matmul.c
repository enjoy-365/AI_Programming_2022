// written by Jongsoo Kim
// Last modification: 2014-06-22
// compile options gcc -std=c99 -O3 matmul.c 

// N1024 : Intel(R) Xeon(R) CPU           E5640  @ 2.67GHz
// elapsed time in seconds = 9.750000e+00
// number of multiplications and additions  = 2.147484e+09
// 2.202547e-01 Gflops

// N=1024, Intel(R) Xeon(R) CPU E5-2660 @ 2.20GHz
// elapsed time in seconds = 2.080000e+00
// number of multiplications and additions  = 2.147484e+09
// 1.032444e+00 Gflops

#include <stdio.h>  
#include <stdlib.h>  // for malloc
#include <time.h>    // for the clock() function

const int N = 1024;

void MatMul(const float * A, const float * B, float * C) {

    for (int row=0; row<N; ++row)  
    for (int col=0; col<N; ++col) { 
        float Cvalue = 0;
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * N + col];
        }
        C[row*N+col] = Cvalue;
    }

}

int main (void) {

    double etime = (double) clock(); // start to measure the elapsed time

    float * A, * B, *C;
    size_t size = N * N * sizeof(float);
    A = (float *) malloc(size);
    B = (float *) malloc(size);
    C = (float *) malloc(size);

    for (int row=0; row<N; ++row)  
    for (int col=0; col<N; ++col) { 
        A[row * N + col] = row + col + 1.0f; 
        B[row * N + col] = row + col + 1.0f; 
    }

    MatMul (A, B, C);

//    for (int row=0; row<N; ++row)  
//    for (int col=0; col<N; ++col) { 
//        printf("%f\n", C[row*N+col]);
//    }

    free(A); free(B); free(C);

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;
    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = 2.0f*N*N*N;
    printf("number of multiplications and additions  = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    return 0;
}
