#include <iostream>
#include "dcl.h"

static inline data_t abs_fp(data_t x) {
    return (x < (data_t)0) ? (data_t)(-x) : x;
}

static inline data_t clamp_fp(data_t x, data_t lo, data_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void init_input(data_t in[N]) {
    for (int k = 0; k < N; k++) {
        int v = ((k * 73 + 19) & 1023) - 512;   // [-512, 511]
        in[k] = (data_t)v / (data_t)256;        // about [-2, 2)
    }
}

static void golden_kernel(const data_t in[N], data_t out[N]) {
    static data_t s0[N];
    static data_t s1[N];
    static stat_t stats[N / BLOCK];
    static data_t s3[N];

    const coef_t alpha = (coef_t)0.875;
    const coef_t beta  = (coef_t)0.125;

    const coef_t w0 = (coef_t)0.50;
    const coef_t w1 = (coef_t)(-0.25);
    const coef_t w2 = (coef_t)0.125;

    const stat_t eps = (stat_t)0.5;

    const coef_t gamma = (coef_t)1.25;
    const coef_t delta = (coef_t)0.05;

    // K0
    for (int k = 0; k < N; k++) {
        s0[k] = (data_t)((acc_t)alpha * (acc_t)in[k] + (acc_t)beta);
    }

    // K1
    for (int k = 0; k < N; k++) {
        data_t x0 = s0[k];
        data_t x1 = (k >= 1) ? s0[k - 1] : (data_t)0;
        data_t x2 = (k >= 2) ? s0[k - 2] : (data_t)0;

        acc_t acc = (acc_t)w0 * (acc_t)x0 + (acc_t)w1 * (acc_t)x1 + (acc_t)w2 * (acc_t)x2;
        data_t y = (data_t)acc;
        y = abs_fp(y);
        y = clamp_fp(y, (data_t)0, (data_t)7.5);
        s1[k] = y;
    }

    // K2
    for (int b = 0; b < (N / BLOCK); b++) {
        acc_t sum_abs = 0;
        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            sum_abs += (acc_t)abs_fp(s0[base + i]);
        }
        stat_t avg_abs = (stat_t)(sum_abs / (acc_t)BLOCK);
        stats[b] = avg_abs + eps;
    }

    // K3 (1 division per block)
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t st = stats[b];
        stat_t inv_st = (stat_t)((acc_t)1 / (acc_t)st);

        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            s3[base + i] = (data_t)((acc_t)s1[base + i] * (acc_t)inv_st);
        }
    }

    // K4
    for (int k = 0; k < N; k++) {
        data_t z = (data_t)((acc_t)gamma * (acc_t)s3[k] + (acc_t)delta);
        z = clamp_fp(z, (data_t)0, (data_t)7.9);
        out[k] = z;
    }
}

int main() {
    static data_t in[N];
    static data_t out_hw[N];
    static data_t out_gold[N];

    init_input(in);

    top_kernel(in, out_hw);
    golden_kernel(in, out_gold);

    int errors = 0;
    for (int k = 0; k < N; k++) {
        if (out_hw[k] != out_gold[k]) {
            errors++;
            if (errors <= 10) {
                std::cout << "Mismatch at k=" << k
                          << " hw=" << out_hw[k].to_double()
                          << " gold=" << out_gold[k].to_double()
                          << "\n";
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
