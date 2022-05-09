
#include "dcl.h"

void matrix_mul( FIX_TYPE MatA_DRAM[M][N], FIX_TYPE MatB_DRAM[N][K], FIX_TYPE MatC_DRAM[M][K]);

int main()
{

    FIX_TYPE MatA_tb[M][N];
    FIX_TYPE MatB_tb[N][K];
    FIX_TYPE MatC_tb[M][K];
    FIX_TYPE MatC_expected[M][K];

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            MatA_tb[i][j] = (FIX_TYPE)(((float)rand() / (float)RAND_MAX - 0.5) * 2);
        }
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            MatB_tb[i][j] = (FIX_TYPE)(((float)rand() / (float)RAND_MAX - 0.5) * 2);
        }
    }

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            MatC_tb[i][j] = 0;
            MatC_expected[i][j] = 0;
        }
    }

    matrix_mul(MatA_tb, MatB_tb, MatC_tb);

    // Expected value for MatC_tb
    // To make sure your optimizations in matrix_mul is not changing the functionality
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            
            for(int p = 0; p < N; p++) {
                MatC_expected[i][j] += MatA_tb[i][p] * MatB_tb[p][j];
            }

        }
    }

    // Verify if the output of matrix_mul is correct
    int passed = 1;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            if( MatC_tb[i][j] != MatC_expected[i][j]) {
                printf("Mismatch at MatC[%d][%d]: expected %.2f, got %.2f\n", i, j, MatC_expected[i][j].to_float(), MatC_tb[i][j].to_float());
                passed = 0;
            }
        }
    }
    if(passed) {
        printf("Your test passed!\n");
    }


    return 0;
}
