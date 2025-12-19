#include "dcl.h"

void top_kernel( DATA_TYPE a[100], DATA_TYPE b[100], DATA_TYPE sum[100])
{
#pragma HLS interface m_axi port=a offset=slave bundle=a
#pragma HLS interface m_axi port=b offset=slave bundle=b
#pragma HLS interface m_axi port=sum offset=slave bundle=sum
#pragma HLS interface s_axilite port=return


	for(int i = 0; i < 100; i++) {
		sum[i] = a[i] * b[i];
	}

}
