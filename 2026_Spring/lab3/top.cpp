#include "dcl.h"

// Baseline: 5-stage DAG written as array passes (correct but slow).
// Conceptual kernels (should refactor into dataflow with hls::stream):
//   K0: preprocess
//   K1: transform (sliding window)
//   K2: per-block statistic (1 token per block, delayed until block complete)  [extra-hard twist]
//   K3: join + normalize using inv_stat (1 division per block, then multiply per element)
//   K4: postprocess + store

static inline data_t abs_fp(data_t x) {
    return (x < (data_t)0) ? (data_t)(-x) : x;
}

static inline data_t clamp_fp(data_t x, data_t lo, data_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void top_kernel(const data_t in[N],
                data_t out[N]) {
    static data_t s0[N];               // after preprocess
    static data_t s1[N];               // after transform
    static stat_t stats[N / BLOCK];    // 1 stat per block
    static data_t s3[N];               // after normalize

    // Coefficients (constants)
    const coef_t alpha = (coef_t)0.875;
    const coef_t beta  = (coef_t)0.125;

    const coef_t w0 = (coef_t)0.50;
    const coef_t w1 = (coef_t)(-0.25);
    const coef_t w2 = (coef_t)0.125;

    const stat_t eps = (stat_t)0.5;    // avoid tiny stats
    const coef_t gamma = (coef_t)1.25;
    const coef_t delta = (coef_t)0.05;

    // -------------------------
    // K0: preprocess
    // -------------------------
    for (int k = 0; k < N; k++) {
        s0[k] = (data_t)((acc_t)alpha * (acc_t)in[k] + (acc_t)beta);
    }

    // -------------------------
    // K1: transform (3-tap + abs + clamp)
    // -------------------------
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

    // -------------------------
    // K2: per-block statistic (delayed)
    // stats[b] = avg_abs(s0[block]) + eps
    // -------------------------
    for (int b = 0; b < (N / BLOCK); b++) {
        acc_t sum_abs = 0;
        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            sum_abs += (acc_t)abs_fp(s0[base + i]);
        }
        stat_t avg_abs = (stat_t)(sum_abs / (acc_t)BLOCK);
        stats[b] = avg_abs + eps;
    }

    // -------------------------
    // K3: join + normalize (1 division per block, multiply per element)
    // s3[k] = s1[k] * inv_stat(block(k))
    // -------------------------
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t st = stats[b];

        // One division per block:
        // inv_stat = 1 / st
        stat_t inv_st = (stat_t)((acc_t)1 / (acc_t)st);

        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            s3[base + i] = (data_t)((acc_t)s1[base + i] * (acc_t)inv_st);
        }
    }

    // -------------------------
    // K4: postprocess + store
    // out[k] = clamp(s3[k] * gamma + delta, 0, 7.9)
    // -------------------------
    for (int k = 0; k < N; k++) {
        data_t z = (data_t)((acc_t)gamma * (acc_t)s3[k] + (acc_t)delta);
        z = clamp_fp(z, (data_t)0, (data_t)7.9);
        out[k] = z;
    }
}
