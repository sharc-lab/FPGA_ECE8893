#include "dcl.h"

void matrix_mul( FIX_TYPE MatA_DRAM[M][N], FIX_TYPE MatB_DRAM[N][K], FIX_TYPE MatC_DRAM[M][K])
{
#pragma HLS interface m_axi depth=100000 port=MatA_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=100000 port=MatB_DRAM offset=slave bundle=mem
#pragma HLS interface m_axi depth=100000 port=MatC_DRAM offset=slave bundle=mem
#pragma HLS interface s_axilite port=return
    
    FIX_TYPE MatA[M][N];
    FIX_TYPE MatB[N][K];
    FIX_TYPE MatC[M][K];

    // Read in the data from DRAM to BRAM
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            MatA[i][j] = MatA_DRAM[i][j];
        }
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            MatB[i][j] = MatB_DRAM[i][j];
        }
    }

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            MatC[i][j] = 0;
        }
    }

    // Compute the matrix multiplication
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            
            for(int p = 0; p < N; p++) {
                MatC[i][j] += MatA[i][p] * MatB[p][j];
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
