#include "dcl.h"

void matrix_mul( FIX_TYPE MatA_DRAM[M][N], FIX_TYPE MatB_DRAM[N][K], FIX_TYPE MatC_DRAM[M][K])
{
	//Separate bundle used for MatA_DRAM, MatB_DRAM and MatC_DRAM.
	#pragma HLS interface m_axi depth=100000 port=MatA_DRAM offset=slave bundle=mem1
	#pragma HLS interface m_axi depth=100000 port=MatB_DRAM offset=slave bundle=mem2
	#pragma HLS interface m_axi depth=100000 port=MatC_DRAM offset=slave bundle=mem3
	#pragma HLS interface s_axilite port=return

    FIX_TYPE MatA[M][N];
    FIX_TYPE MatB[N][K];
    FIX_TYPE MatC[M][K];

    ////Read in the data from DRAM to BRAM and initialize MatC. Read loops combined
    int bound_a = (M < N)? N: M;
    int bound_b = (N < K)? K: N;

    //Outer loop flattening is disabled. As per Vitis documentation loop flattening should be avoided for reading from or writing to DRAM through AXI
    outer_read: for(int i = 0; i < bound_a; i++)
	#pragma HLS LOOP_FLATTEN off
    	inner_read: for(int j = 0; j < bound_b; j++) {
        	if (i < M && j < N)
        		MatA[i][j] = MatA_DRAM[i][j];
        	if (i < N && j < K)
        		MatB[i][j] = MatB_DRAM[i][j];
        	if (i < M && j < K)
        		MatC[i][j] = 0;
        }

	#pragma HLS ARRAY_PARTITION variable=MatC dim=2 factor=75 cyclic
	#pragma HLS ARRAY_PARTITION variable=MatB dim=2 factor=75 cyclic
    // Compute the matrix multiplication. Loop reordering + loop tiling with tile size 75 + array partitioning A & B along columns.
    L1: for(int i = 0; i < M; i++)
    	L2: for(int p = 0; p < N; p++)
    		L3: for(int j = 0; j < K; j+=75)
			#pragma HLS PIPELINE
    			L4: for(int jj = 0; jj < 75; jj++)
				#pragma HLS UNROLL
    				MatC[i][j+jj] = MatA[i][p] * MatB[p][j+jj];

    //Write back the data from BRAM to DRAM.
    outer_write: for(int i = 0; i < M; i++)
	#pragma HLS LOOP_FLATTEN off
    	inner_write: for(int j = 0; j < K; j++)
		#pragma HLS PIPELINE
    		MatC_DRAM[i][j] = MatC[i][j];

}


