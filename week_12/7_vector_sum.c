// written by Jongsoo Kim
// Last modification: 2014-06-04
// compile options gcc -std=c99 vector_sum.c 

#include <stdio.h>

const int N = 128;

void add(int *a, int *b, int *c) { 
    int i = 0;
    while (i < N) {
        c[i] = a[i] + b[i];
        i += 1;
    }
}
     
int main (void) {
    int a[N], b[N], c[N];

    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    add (a, b, c);

    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    return 0;
}
