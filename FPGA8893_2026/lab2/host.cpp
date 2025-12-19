#include <iostream>
#include <cstdlib>
#include "dcl.h"

static void init_input(data_t A[NX][NY]) {
    // Deterministic pattern in fixed-point
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int v = (i * 17 + j * 31) % 1024;   // 0..1023
            // Map to roughly [0, 1) in fixed-point without floats:
            // A = v / 1024
            A[i][j] = (data_t)v * (data_t)(1.0 / 1024.0); // constant is compile-time; stored as fixed
        }
    }
}

// Golden model in fixed-point (bit-exact target)
static void golden_kernel(const data_t A_in[NX][NY],
                          data_t A_out[NX][NY]) {
    static data_t cur[NX][NY];
    static data_t nxt[NX][NY];

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    // Copy input
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            cur[i][j] = A_in[i][j];
        }
    }

    for (int t = 0; t < TSTEPS; t++) {
        // Boundaries unchanged
        for (int j = 0; j < NY; j++) {
            nxt[0][j]      = cur[0][j];
            nxt[NX - 1][j] = cur[NX - 1][j];
        }
        for (int i = 0; i < NX; i++) {
            nxt[i][0]      = cur[i][0];
            nxt[i][NY - 1] = cur[i][NY - 1];
        }

        // Interior update
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                acc_t sum_axis =
                    (acc_t)cur[i - 1][j] + (acc_t)cur[i + 1][j] +
                    (acc_t)cur[i][j - 1] + (acc_t)cur[i][j + 1];

                acc_t sum_diag =
                    (acc_t)cur[i - 1][j - 1] + (acc_t)cur[i - 1][j + 1] +
                    (acc_t)cur[i + 1][j - 1] + (acc_t)cur[i + 1][j + 1];

                acc_t center = (acc_t)cur[i][j];
                acc_t out = (acc_t)wc * center + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag;
                nxt[i][j] = (data_t)out;
            }
        }

        // Swap by copy
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                cur[i][j] = nxt[i][j];
            }
        }
    }

    // Output
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            A_out[i][j] = cur[i][j];
        }
    }
}

int main() {
    static data_t A_in[NX][NY];
    static data_t A_hw[NX][NY];
    static data_t A_gold[NX][NY];

    init_input(A_in);

    top_kernel(A_in, A_hw);
    golden_kernel(A_in, A_gold);

    int errors = 0;

    // Exact match check (fixed-point should be deterministic if algorithm matches)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            if (A_hw[i][j] != A_gold[i][j]) {
                errors++;
                if (errors <= 10) {
                    std::cout << "Mismatch at (" << i << "," << j << "): "
                              << "hw=" << A_hw[i][j]
                              << " gold=" << A_gold[i][j] << "\n";
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << "TEST PASSED\n";
        return 0;
    } else {
        std::cout << "TEST FAILED with " << errors << " mismatches\n";
        return 1;
    }
}
