// written by Jongsoo Kim
// last modification: 2011-06-28
// compile options gcc -std=c99 dot_product.c 

#include <stdio.h>
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 1024;

float dot_product(float *a, float *b) { 

    float c = 0.0f;
    for (int i=0; i<N; i++)
        c += a[i]*b[i];
    return c;
}

int main (void) {

    float a[N], b[N];

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    float c = dot_product(a,b);

    printf("Dot prodoct of a and b = %f\n", c);
    printf("sum_squares of (N-1) = %f\n", sum_squares((float)(N-1)) );

    return 0;
}
