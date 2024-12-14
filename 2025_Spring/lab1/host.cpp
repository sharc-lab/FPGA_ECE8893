
/* This is the host code for the HLSKernel, used to control, provide inputs, and collect results from HLSKernel */
/* Also known as the testbench, used to check the functionality of HLSKernel */

#include "dcl.h"


int main()
{
	DATA_TYPE a[100];
	DATA_TYPE b[100];
	DATA_TYPE c_HLS[100];	// computed by HLSKernel on the FPGA
	DATA_TYPE c_golden[100]; // computed by the host code as a golden reference

	for(int i = 0; i < 100; i++) {
		a[i] = i;
		b[i] = i + 1;
		c_HLS[i] = 0; 
		c_golden[i] = a[i] * b[i];
	}

	// Start HLSKernel on FPGA
	HLSKernel(a, b, c_HLS);

	// Collect results and verify the functionality
	for(int j = 0; j < 100; j++) {
		printf("HLS: %d\tRef: %d\n", c_HLS[j], c_golden[j]);
		//printf("HLS: %.2f\tRef: %.2f\n", c_HLS[j], c_golden[j]);
	}
	printf("\n");

	return 0;
}
