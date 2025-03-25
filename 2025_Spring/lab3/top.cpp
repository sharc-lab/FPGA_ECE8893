#include "dcl.h"


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
{
    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

    #pragma HLS interface m_axi port=C offset=slave bundle=mem3

    #pragma HLS interface s_axilite port=return
    // Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
    zero_out_C: for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = 0.0f;
        }
    }
    Step_1_For_SpM: for (int A_row = 0; A_row < N; A_row++) {
        int index_for_A = row_ptr_A[A_row];
        int ending_index_for_A = row_ptr_A[A_row + 1];
        // Step 2, Loop CSR matrix B along its Column
        For_Step_2_For_SpM: for (int B_column = 0; B_column < M; B_column++) {
            int index_for_B = col_ptr_B[B_column];
            int ending_index_for_B = col_ptr_B[B_column + 1];
            
            // Step 3, Loop through the elements in the row of A and the column of B
            int idx_A = index_for_A;
            int idx_B = index_for_B;
            AB_Iteration_For_SpM: while (idx_A < ending_index_for_A && idx_B < ending_index_for_B) {
                if(column_indices_A[idx_A] == row_indices_B[idx_B]){
                    C[A_row][B_column] += values_A[idx_A] * values_B[idx_B];
                    idx_A=idx_A+1;
                    idx_B=idx_B+1;
                }
                else if(column_indices_A[idx_A] < row_indices_B[idx_B]){
                    idx_A=idx_A+1;
                }
                else{
                    idx_B=idx_B+1;
                }
            }
        }
    }
}
