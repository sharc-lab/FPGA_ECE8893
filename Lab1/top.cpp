#include "dcl.h"

void matrix_mul( FIX_TYPE MatA_DRAM[M][N], FIX_TYPE MatB_DRAM[N][K], FIX_TYPE MatC_DRAM[M][K])
{

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
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < M; j++) {
            
            for(int p = 0; p < N; p++) {
                MatC[i][j] += MatA[j][p] * MatB[p][i];
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