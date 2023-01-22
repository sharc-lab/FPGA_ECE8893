///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    main.cpp
// Description: Test bench for complex matrix multiplication
//
// Note:        You are not required to turn in this code. 
//              So in case you make any modifications (for debugging, etc.),
//              do ensure your design works with the original test bench.
///////////////////////////////////////////////////////////////////////////////

#include "complex.h"

complex_t MatA_tb[M][N];
complex_t MatB_tb[N][K];
complex_t MatC_tb[M][K];
complex_t MatC_expected[M][K];

void loadMatrix(string filename, int numRows, int numCols)
{
    complex_t Mat_tb[numRows][numCols];

    int i = 0, j = 0, t = 0;
    int_t real, imag;

    std::ifstream Mat(filename);

    while(Mat >> real >> imag)
    {
        i = t / numCols;
        j = t % numCols;
        
        if(filename == "MatA.txt")
        {
            MatA_tb[i][j].real = real;
            MatA_tb[i][j].imag = imag;
        }
        else if(filename == "MatB.txt")
        {
            MatB_tb[i][j].real = real;
            MatB_tb[i][j].imag = imag;
        }
        else
        {
            MatC_expected[i][j].real = real;
            MatC_expected[i][j].imag = imag;
        }
        
        t++;
    }
}

int main()
{
    int_t real, imag; 
    int i, j, t;
    
    // Load input matrices and expected output matrix
    loadMatrix("MatA.txt", M, N);
    loadMatrix("MatB.txt", N, K);
    loadMatrix("MatC.txt", M, K);

    // Uncomment for debugging
    //printf("A[0][0] = %d + %dj\n", MatA_tb[0][0].real, MatA_tb[0][0].imag);
    //printf("A[%d][%d] = %d + %dj\n", M-1, N-1, MatA_tb[M-1][N-1].real, MatA_tb[M-1][N-1].imag);
    //printf("B[0][0] = %d + %dj\n", MatB_tb[0][0].real, MatB_tb[0][0].imag);
    //printf("B[%d][%d] = %d + %dj\n", N-1, K-1, MatB_tb[N-1][K-1].real, MatB_tb[N-1][K-1].imag);
    //printf("C[0][0] = %hi + %hij\n", MatC_expected[0][0].real, MatC_expected[0][0].imag);
    //printf("C[%d][%d] = %hi + %hij\n", M-1, K-1, MatC_expected[M-1][K-1].real, MatC_expected[M-1][K-1].imag);
    
    // Call DUT
    complex_matmul(MatA_tb, MatB_tb, MatC_tb);

    // Verify funtional correctness
    int passed = 1;
    for(int i = 0; i < M; i++) 
    {
        for(int j = 0; j < K; j++) 
        {
            if((MatC_tb[i][j].real != MatC_expected[i][j].real) || (MatC_tb[i][j].imag != MatC_expected[i][j].imag)) 
            {
                printf("Mismatch at MatC[%d][%d]: Expected: (%hi + %hij) \t Actual: (%hi + %hij)\n", i, j, MatC_expected[i][j].real, MatC_expected[i][j].imag, MatC_tb[i][j].real, MatC_tb[i][j].imag);
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
