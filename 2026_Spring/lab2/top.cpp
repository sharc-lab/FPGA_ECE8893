#include "dcl.h"

// 9-point weighted stencil (constant weights):
// nxt[i][j] = wc * cur[i][j]
//           + wa * (cur[i-1][j] + cur[i+1][j] + cur[i][j-1] + cur[i][j+1])
//           + wd * (cur[i-1][j-1] + cur[i-1][j+1] + cur[i+1][j-1] + cur[i+1][j+1])
//
// Boundary handling: boundaries are copied unchanged each timestep.

void top_kernel(const data_t A_in[NX][NY],
                data_t A_out[NX][NY]) {
    static data_t cur[NX][NY];
    static data_t nxt[NX][NY];

    // Constant weights (sum to 1.0)
    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    // Copy input into cur
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            cur[i][j] = A_in[i][j];
        }
    }

    // Time stepping
    for (int t = 0; t < TSTEPS; t++) {
        // Copy boundaries unchanged
        for (int j = 0; j < NY; j++) {
            nxt[0][j]      = cur[0][j];
            nxt[NX - 1][j] = cur[NX - 1][j];
        }
        for (int i = 0; i < NX; i++) {
            nxt[i][0]      = cur[i][0];
            nxt[i][NY - 1] = cur[i][NY - 1];
        }

        // Update interior
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

        // Baseline swap: full copy nxt -> cur
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                cur[i][j] = nxt[i][j];
            }
        }
    }

    // Write output
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            A_out[i][j] = cur[i][j];
        }
    }
}
