//####################### Solution by Parima Mehta ###########################//
#include "dcl.h"

#define D 25
#define W 10

void matrix_mul(FIX_TYPE MatA_DRAM[M][N], FIX_TYPE MatB_DRAM[N][K], FIX_TYPE MatC_DRAM[M][K])
{
#pragma HLS interface m_axi depth=100000 port=MatA_DRAM offset=slave bundle=memA
#pragma HLS interface m_axi depth=100000 port=MatB_DRAM offset=slave bundle=memB
#pragma HLS interface m_axi depth=100000 port=MatC_DRAM offset=slave bundle=memC
#pragma HLS interface s_axilite port=return

    FIX_TYPE MatA[M][N];
    FIX_TYPE MatB[N][K];
    FIX_TYPE MatC[M][K];

    // Read in the data from DRAM to BRAM
    int max_m_n = ((M >= N) ? M : N);
    int max_n_k = ((N >= K) ? N : K);
    for(int i = 0; i < max_m_n; i++)
    {
        for(int j = 0; j < max_n_k; j++)
        {
            if((i < M) && (j < N))
            {
                MatA[i][j] = MatA_DRAM[i][j];
            }

            if((i < N) && (j < K))
            {
                MatB[i][j] = MatB_DRAM[i][j];
            }

            if((i < M) && (j < K))
            {
                MatC[i][j] = 0;
            }
        }
    }

#pragma HLS array_partition variable=MatA dim=1 factor=25 cyclic
#pragma HLS array_partition variable=MatB dim=2 factor=10 cyclic
#pragma HLS array_partition variable=MatC dim=1 factor=25 cyclic
#pragma HLS array_partition variable=MatC dim=2 factor=10 cyclic
    // Compute the matrix multiplication
    for(int i = 0; i < M; i += D) 
    {
        for(int p = 0; p < N; p++) 
        {
            for(int j = 0; j < K; j += W)
            {
                #pragma HLS PIPELINE II=1
                for(int ii = 0; ii < D; ii++)
                {
                    FIX_TYPE a = MatA[i + ii][p];
                    for(int jj = 0; jj < W; jj++) 
                    {
                        MatC[i + ii][j + jj] += a * MatB[p][j + jj];
                    }
                }
            }
        }
    }

    // Write back the data from BRAM to DRAM
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            MatC_DRAM[i][j] = MatC[i][j];
        }
    }

}
