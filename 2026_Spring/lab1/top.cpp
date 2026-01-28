#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A
#pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C
#pragma HLS interface s_axilite port=return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];

    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;

        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;

        // Normalize each element in the row
        for (int j = 0; j < N_COLS; j++) {
            tmp[i][j] = A[i][j] / denom;
        }
    }

    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++) {
        data_t col_sum = 0.0;

        // Compute column sum of normalized values
        for (int i = 0; i < N_ROWS; i++) {
            col_sum += tmp[i][j];
        }

        // Compute average as scale
        data_t scale = col_sum / (data_t)N_ROWS;

        // Apply scale to each element in the column
        for (int i = 0; i < N_ROWS; i++) {
            C[i][j] = tmp[i][j] * scale;
        }
    }
    
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            C_DRAM[i][j] = C[i][j];
        }
    }
}
