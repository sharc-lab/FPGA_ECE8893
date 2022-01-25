#include <stdio.h>

void top( int a[100], int b[100], int sum[100]);

int main()
{
        int a[100];
        int b[100];
        int c[100];

        for(int i = 0; i < 100; i++) {
                a[i] = i; b[i] = i * 2; c[i] = 0;
        }

		// call the function, i.e., your adder
        top(a, b, c);

		// verify the results
        for(int j = 0; j < 100; j++) {
                printf("%d\n", c[j]);
        }
        return 0;
}
