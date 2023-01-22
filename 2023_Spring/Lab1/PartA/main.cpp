///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    main.cpp
// Description: Test bench for real matrix multiplication
//
// Note:        You are not required to turn in this code. 
//              So in case you make any modifications (for debugging, etc.),
//              do ensure your design works with the original test bench.
///////////////////////////////////////////////////////////////////////////////

#include "real.h"

int main()
{
    // Declare matrices
    real_t MatA_tb[M][N];
    real_t MatB_tb[N][K];
    real_t MatC_tb[M][K];
    real_t MatC_expected[M][K];

    // Generate Matrix A with random values
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            MatA_tb[i][j] = rand() % 50;
        }
    }

    // Generate Matrix B with random values
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            MatB_tb[i][j] = rand() % 50;
        }
    }

    // Initialize Matrix C 
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            MatC_tb[i][j] = 0;
        }
    }

    // Call DUT
    real_matmul(MatA_tb, MatB_tb, MatC_tb);

    // Expected value for Matrix C
    // To make sure your optimizations do not change the functionality
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            
            MatC_expected[i][j] = 0;
            for(int p = 0; p < N; p++) {
                MatC_expected[i][j] += MatA_tb[i][p] * MatB_tb[p][j];
            }

        }
    }

    // Verify functional correctness before synthesizing
    int passed = 1;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            if(MatC_tb[i][j] != MatC_expected[i][j]) {
                printf("Mismatch at MatC[%d][%d]: Expected: %hi \t Actual: %hi\n", 
                        i, j, MatC_expected[i][j], MatC_tb[i][j]);
                passed = 0;
            }
        }
    }
    
    if(passed) {
        printf("-----------------------------------\n");
        printf("|         TEST PASSED!            |\n");
        printf("-----------------------------------\n");
    }
    else {
        printf("-----------------------------------\n");
        printf("|         TEST FAILED :(          |\n");
        printf("-----------------------------------\n");
    }

    return 0;
}
