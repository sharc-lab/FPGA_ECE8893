#include "dcl.h"

// 9-point weighted stencil (constant weights):
// nxt[i][j] = wc * cur[i][j]
//           + wa * (cur[i-1][j] + cur[i+1][j] + cur[i][j-1] + cur[i][j+1])
//           + wd * (cur[i-1][j-1] + cur[i-1][j+1] + cur[i+1][j-1] + cur[i+1][j+1])
//
// Boundary handling: boundaries are copied unchanged each timestep.

void top_kernel(const data_t A_in[NX][NY],
                data_t A_out[NX][NY]) {
    static data_t buf0[NX][NY];
    static data_t buf1[NX][NY];
#pragma HLS ARRAY_PARTITION variable=buf0 cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=buf1 cyclic factor=3 dim=1

    // Constant weights (sum to 1.0)
    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    // Copy input into initial source buffer.
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
#pragma HLS PIPELINE II=1
            buf0[i][j] = A_in[i][j];
        }
    }

    bool ping = true;

    // Time stepping with ping-pong buffering.
    for (int t = 0; t < TSTEPS; t++) {
        data_t (*src)[NY] = ping ? buf0 : buf1;
        data_t (*dst)[NY] = ping ? buf1 : buf0;

        // Copy boundaries unchanged.
        for (int j = 0; j < NY; j++) {
#pragma HLS PIPELINE II=1
            dst[0][j]      = src[0][j];
            dst[NX - 1][j] = src[NX - 1][j];
        }
        for (int i = 0; i < NX; i++) {
#pragma HLS PIPELINE II=1
            dst[i][0]      = src[i][0];
            dst[i][NY - 1] = src[i][NY - 1];
        }

        // Update interior with a sliding 3x3 window across columns.
        for (int i = 1; i < NX - 1; i++) {
            data_t t0 = src[i - 1][0];
            data_t t1 = src[i - 1][1];
            data_t t2 = src[i - 1][2];
            data_t m0 = src[i][0];
            data_t m1 = src[i][1];
            data_t m2 = src[i][2];
            data_t b0 = src[i + 1][0];
            data_t b1 = src[i + 1][1];
            data_t b2 = src[i + 1][2];

            for (int j = 1; j < NY - 1; j++) {
#pragma HLS PIPELINE II=1
                acc_t sum_axis =
                    (acc_t)t1 + (acc_t)b1 +
                    (acc_t)m0 + (acc_t)m2;

                acc_t sum_diag =
                    (acc_t)t0 + (acc_t)t2 +
                    (acc_t)b0 + (acc_t)b2;

                acc_t center = (acc_t)m1;

                acc_t out = (acc_t)wc * center + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag;
                dst[i][j] = (data_t)out;

                if (j < NY - 2) {
                    t0 = t1;
                    t1 = t2;
                    t2 = src[i - 1][j + 2];

                    m0 = m1;
                    m1 = m2;
                    m2 = src[i][j + 2];

                    b0 = b1;
                    b1 = b2;
                    b2 = src[i + 1][j + 2];
                }
            }
        }

        ping = !ping;
    }

    data_t (*final_buf)[NY] = ping ? buf0 : buf1;

    // Write output.
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
#pragma HLS PIPELINE II=1
            A_out[i][j] = final_buf[i][j];
        }
    }
}
