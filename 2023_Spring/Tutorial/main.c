// main.c
#include <stdio.h>

void top( int a[100], int b[100], int sum[100]);

int main()
{
		int a[100];
		int b[100];
		int c[100];

		for(int i = 0; i < 100; i++) 
		{
			a[i] = i; 
			b[i] = i * 2; 
			c[i] = 0;
    }

		// Call the DUT function, i.e., your adder
    top(a, b, c);

		// verify the results
		bool pass = 1;
    for(int j = 0; j < 100; j++) 
		{
		    if(c[j] != (a[j] + b[j]))
				{
						pass = 0;
				}
				printf("A[%d] = %d; B[%d] = %d; Sum C[%d] = %d\n", j, a[j], j, b[j], j, c[j]); 
    }

		if(pass)
    	printf("Test Passed! :) \n");
		else
			printf("Test Failed :( \n");

    return 0;
}
