#include <iostream>
#include <cstdlib>
#include <cmath>

#include "dcl.h"

static void init_input(data_t A[N_ROWS][N_COLS]) {
    // Deterministic pseudo-random-ish pattern
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            // Example pattern: depends on i and j but fixed
            int val = (i * 17 + j * 31) % 100;
            A[i][j] = (data_t)val / (data_t)10.0; // range ~[0, 9.9]
        }
    }
}

// Golden CPU reference implementation (same algorithm as top_kernel)
static void golden_kernel(data_t A[N_ROWS][N_COLS],
                          data_t C[N_ROWS][N_COLS]) {
    data_t tmp[N_ROWS][N_COLS];

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;
        for (int j = 0; j < N_COLS; j++) {
            row_sum += A[i][j];
        }
        data_t denom = row_sum + (data_t)1.0;
        for (int j = 0; j < N_COLS; j++) {
            tmp[i][j] = A[i][j] / denom;
        }
    }

    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++) {
        data_t col_sum = 0.0;
        for (int i = 0; i < N_ROWS; i++) {
            col_sum += tmp[i][j];
        }
        data_t scale = col_sum / (data_t)N_ROWS;
        for (int i = 0; i < N_ROWS; i++) {
            C[i][j] = tmp[i][j] * scale;
        }
    }
}

int main() {
    static data_t A[N_ROWS][N_COLS];
    static data_t C_hw[N_ROWS][N_COLS];
    static data_t C_gold[N_ROWS][N_COLS];

    init_input(A);

    // Run hardware (HLS) version
    top_kernel(A, C_hw);

    // Run golden software version
    golden_kernel(A, C_gold);

    // Compare results
    int errors = 0;
    const float tol = 1e-3f;

    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            if (C_hw[i][j] != C_gold[i][j]) {
                errors++;
                if (errors < 10) {
                    std::cout << "Mismatch at (" << i << ", " << j << "): "
                              << "C_hw=" << C_hw[i][j].to_float()
                              << ", C_gold=" << C_gold[i][j].to_float() << std::endl;
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << "TEST PASSED!!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED with " << errors << " mismatches." << std::endl;
        return 1;
    }
}
