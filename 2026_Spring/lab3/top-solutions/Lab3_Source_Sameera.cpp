#include "dcl.h"
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Baseline: 5-stage DAG written as array passes (correct but slow).
// Conceptual kernels (should refactor into dataflow with hls::stream):
//   K0: preprocess
//   K1: transform (sliding window)
//   K2: per-block statistic (1 token per block, delayed until block complete)  [extra-hard twist]
//   K3: join + normalize using inv_stat (1 division per block, then multiply per element)
//   K4: postprocess + store

#ifndef FACTOR
#define FACTOR 32
#endif

#ifndef MOD_DATA_BITS_W
#define MOD_DATA_BITS_W 17
#endif

#ifndef DATA_INT_BITS_W
#define DATA_INT_BITS_W 2
#endif

static const int IN_DATA_BITS  = 32;   // 32 (from dcl.h)
static const int MOD_DATA_BITS = MOD_DATA_BITS_W; // reduced internal storage width
static const int DATA_INT_BITS = DATA_INT_BITS_W; // keep same integer range as data_t
static const int DATA_FRAC_BITS = MOD_DATA_BITS - DATA_INT_BITS;
static const int WIDE_BITS     = IN_DATA_BITS * FACTOR;
static const int CHUNKS_PER_BLOCK = BLOCK / FACTOR;
static const int LANE_GROUPS = FACTOR / 4;

static_assert((BLOCK % FACTOR) == 0, "BLOCK must be divisible by FACTOR");
static_assert((FACTOR % 4) == 0, "FACTOR must be divisible by 4");

typedef ap_uint<WIDE_BITS> wide_dt;
typedef ap_uint<MOD_DATA_BITS * FACTOR> stype_t;
typedef ap_int<MOD_DATA_BITS> raw_fixed_t;
typedef ap_fixed<MOD_DATA_BITS, DATA_INT_BITS, AP_TRN, AP_WRAP> modified_stat_t;
typedef ap_fixed<MOD_DATA_BITS, DATA_INT_BITS, AP_TRN, AP_WRAP> modified_coef_t;
typedef ap_fixed<MOD_DATA_BITS, DATA_INT_BITS, AP_TRN, AP_WRAP> modified_data_t;
typedef ap_fixed<19, 9, AP_TRN, AP_WRAP> modified_acc_t;   // accumulator for reductions


struct wide_bus_t {
    data_t elements[FACTOR];
};


#define RANGE(idx, WIDTH) range((idx * WIDTH) + (WIDTH - 1), idx * WIDTH)


static inline modified_data_t abs_fp(modified_data_t x) {
    return (x < (modified_data_t)0) ? (modified_data_t)(-x) : x;
}


static inline modified_data_t clamp_fp(modified_data_t x, modified_data_t lo, modified_data_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline raw_fixed_t raw_from_bits(ap_uint<MOD_DATA_BITS> bits) {
#pragma HLS INLINE
    raw_fixed_t raw = 0;
    raw.range(MOD_DATA_BITS - 1, 0) = bits;
    return raw;
}

static inline raw_fixed_t raw_from_fixed(modified_data_t value) {
#pragma HLS INLINE
    raw_fixed_t raw = 0;
    raw.range(MOD_DATA_BITS - 1, 0) = value.range(MOD_DATA_BITS - 1, 0);
    return raw;
}

static inline raw_fixed_t mul_q15(raw_fixed_t lhs, raw_fixed_t rhs) {
#pragma HLS INLINE
    ap_int<2 * MOD_DATA_BITS> product = (ap_int<2 * MOD_DATA_BITS>)lhs * (ap_int<2 * MOD_DATA_BITS>)rhs;
#pragma HLS bind_op variable=product op=mul impl=dsp latency=4
    ap_int<2 * MOD_DATA_BITS> shifted = product >> DATA_FRAC_BITS;
    raw_fixed_t result = 0;
    result.range(MOD_DATA_BITS - 1, 0) = shifted.range(MOD_DATA_BITS - 1, 0);
    return result;
}


static void stream_from_dram(const data_t in[N], hls::stream<wide_dt>& out) {
#pragma HLS INLINE off
    const wide_dt (*A_wide) = reinterpret_cast<const wide_dt *>(in);
    for (int k = 0; k < N / FACTOR; k++) {
#pragma HLS PIPELINE II = 1
        wide_dt val = A_wide[k];
        out.write(val);
    }
}


static void func_k0(hls::stream<wide_dt>& in, hls::stream<stype_t>& out_1, hls::stream<stype_t>& out_2) {
#pragma HLS INLINE off
    // Coefficients (constants)
    // const modified_coef_t alpha = (modified_coef_t)0.875;
    const modified_coef_t beta  = (modified_coef_t)0.125;
    // -------------------------
    // K0: preprocess
    // -------------------------
    k0_loop: for (int k = 0; k < N / FACTOR; k++) {
#pragma HLS PIPELINE II = 1
        wide_dt in_data = in.read();
        // Use a union to "cast" the bits without logic.
        // Initialize raw at declaration to avoid default-constructing the union.
        union {
            wide_dt raw;
            wide_bus_t bus;
        } converter = {in_data};
#pragma HLS ARRAY_PARTITION variable=converter.bus.elements type=complete dim=1

        stype_t s0data;
        for(int jj=0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
            modified_data_t unpacked = (modified_data_t)converter.bus.elements[jj];
            modified_data_t preprocessed_data = unpacked - (unpacked >> 3);
            preprocessed_data += beta;
            ap_uint<MOD_DATA_BITS> bits = preprocessed_data.range(MOD_DATA_BITS - 1, 0);
            s0data.RANGE(jj, MOD_DATA_BITS) = bits;
        }
        out_1.write(s0data);
        out_2.write(s0data);
    }
}

static void func_k1(hls::stream<stype_t>& in, hls::stream<stype_t>& out) {
#pragma HLS INLINE off
    // w0 = 0.5  kept as multiply (not a clean shift on ap_fixed)
    // w1 = -0.25  => right-shift 2 + negate  (no DSP)
    // w2 = 0.125 => right-shift 3 (no DSP)
    const modified_coef_t w0 = (modified_coef_t)0.50;

    // History across packet boundaries
    modified_data_t hist1 = 0;   // x[k-1]
    modified_data_t hist2 = 0;   // x[k-2]

    modified_data_t ext[FACTOR + 2];
#pragma HLS bind_storage variable=ext type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=ext type=complete dim=0

k1_loop:
    for (int k = 0; k < N / FACTOR; k++) {
#pragma HLS PIPELINE II=1
        stype_t in_data = in.read();

        // --- 1. Unpack directly into ext[2..FACTOR+1] ---
        for (int jj = 0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
            modified_data_t sample;
            sample.range(MOD_DATA_BITS - 1, 0) = in_data.RANGE(jj, MOD_DATA_BITS);
            ext[jj + 2] = sample;
        }

        // --- 2. Prepend history (dependency-free) ---
        ext[0] = hist2;
        ext[1] = hist1;

        // --- 3. FIR + abs + clamp — all in one unrolled pass ---
        stype_t s1data = (stype_t)0;
        for (int jj = 0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
            modified_data_t x0 = ext[jj + 2];   // current
            modified_data_t x1 = ext[jj + 1];   // x[k-1]
            modified_data_t x2 = ext[jj + 0];   // x[k-2]

            // w0*x0  uses one DSP
            // w1*x1 = -x1/4  => arithmetic shift, zero DSPs
            // w2*x2 =  x2/8  => arithmetic shift, zero DSPs
            modified_acc_t acc = (x0 >> 1) - (x1 >> 2) + (x2 >> 3);

            // Saturating abs+clamp in a single expression:
            //   result = acc < 0 ? -acc : acc, then cap at 7.5
            // Merged into one conditional chain — one MUX, not two.
            modified_data_t y = (modified_data_t)acc;
            y = abs_fp(y);
            // y = clamp_fp(y, (modified_data_t)0, (modified_data_t)7.5); NOT REQUIRED
            s1data.RANGE(jj, MOD_DATA_BITS) = y.range((MOD_DATA_BITS - 1), 0);
        }
        out.write(s1data);

        // --- 4. History update directly from ext[] (already in registers) ---
        hist2 = ext[FACTOR];      // was curr[FACTOR-2]
        hist1 = ext[FACTOR + 1];  // was curr[FACTOR-1]
    }
}

// input = s0_2, output = stats
static void func_k2(hls::stream<stype_t>& in, hls::stream<modified_stat_t>& out) {
#pragma HLS INLINE off
    // -------------------------
    // K2: per-block statistic (delayed)
    // stats[b] = avg_abs(s0[block]) + eps
    // -------------------------
    const modified_stat_t eps = (modified_stat_t)0.5;
    for (int b = 0; b < (N / BLOCK); b++) {
        modified_acc_t sum_abs = 0;
        for (int i = 0; i < CHUNKS_PER_BLOCK; i++) {
#pragma HLS PIPELINE II = 1
            stype_t s0data = in.read();
            modified_acc_t sum_local[LANE_GROUPS];
#pragma HLS array_partition variable=sum_local type=complete dim=0
            for (int jj = 0; jj < FACTOR; jj+=4) {
#pragma HLS UNROLL
                //adder tree
                modified_data_t sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
                sum1.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE(jj, MOD_DATA_BITS); 
                sum2.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 1), MOD_DATA_BITS); 
                sum3.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 2), MOD_DATA_BITS);
                sum4.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 3), MOD_DATA_BITS);
                // sum5.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 4), MOD_DATA_BITS);
                // sum6.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 5), MOD_DATA_BITS);
                // sum7.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 6), MOD_DATA_BITS);
                // sum8.range((MOD_DATA_BITS - 1), 0) = s0data.RANGE((jj + 7), MOD_DATA_BITS);

                sum1 = abs_fp(sum1);
                sum2 = abs_fp(sum2);
                sum3 = abs_fp(sum3);
                sum4 = abs_fp(sum4);
                // sum5 = abs_fp(sum5);
                // sum6 = abs_fp(sum6);
                // sum7 = abs_fp(sum7);
                // sum8 = abs_fp(sum8);

                sum_local[jj / 4] = (modified_acc_t)sum1 + (modified_acc_t)sum2 + (modified_acc_t)sum3 + (modified_acc_t)sum4;// + \
                                        // (modified_acc_t)sum5 + (modified_acc_t)sum6 + (modified_acc_t)sum7 + (modified_acc_t)sum8;
#pragma HLS bind_op variable=sum_local op=add impl=fabric latency=1
            }
            modified_acc_t chunk_sum = 0;
            for (int group = 0; group < LANE_GROUPS; group++) {
#pragma HLS UNROLL
                chunk_sum += sum_local[group];
            }
            sum_abs += chunk_sum;
        }
        modified_stat_t avg_abs = sum_abs >> 8;
        modified_stat_t stats_local = avg_abs + eps;
        out.write(stats_local);
    }
}

static void func_sub(hls::stream<modified_stat_t>& stats, hls::stream<modified_stat_t>& inv_stats) {
#pragma HLS INLINE off
    // Constants in your fixed-point type
    const modified_stat_t two = 2.0;
    loop_sub: for (int b = 0; b < (N / BLOCK); b++) {
#pragma HLS PIPELINE II=1
        modified_stat_t D = stats.read();
        // modified_stat_t x = (modified_stat_t)0.75;
        // modified_stat_t mult1_step1 = D * x;
        modified_stat_t mult1_step1 = D - (D >> 2);
// #pragma HLS bind_op variable=mult1_step1 op=mul impl=dsp latency=3
        modified_stat_t sub1 = two - mult1_step1;
        modified_stat_t x1 = sub1 - (sub1 >> 2);
// #pragma HLS bind_op variable=x1 op=mul impl=dsp latency=3
        inv_stats.write(x1);
    }
}

// -------------------------
// K3: join + normalize (1 division per block, multiply per element)
// s3[k] = s1[k] * inv_stat(block(k))
// -------------------------
static void func_k3(hls::stream<modified_stat_t>& inv_stats, hls::stream<stype_t>& s1, hls::stream<stype_t>& s3) {
#pragma HLS INLINE off
    loop_k3: for (int b = 0; b < (N / BLOCK); b++) {
        raw_fixed_t local = raw_from_fixed(inv_stats.read());
        raw_fixed_t local_lanes[FACTOR];
#pragma HLS ARRAY_PARTITION variable=local_lanes type=complete
        for (int jj = 0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
            local_lanes[jj] = local;
        }
        for (int ii = 0; ii < CHUNKS_PER_BLOCK; ii++) {
#pragma HLS PIPELINE II = 1
            stype_t s1data = s1.read();
            stype_t s3data = (stype_t)0;
            // unpack & process
            for (int jj = 0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
                ap_uint<MOD_DATA_BITS> in_bits = s1data.RANGE(jj, MOD_DATA_BITS);
                raw_fixed_t in_data = raw_from_bits(in_bits);
                raw_fixed_t y = mul_q15(in_data, local_lanes[jj]);
                modified_data_t y_fp;
                y_fp.range(MOD_DATA_BITS - 1, 0) = y.range(MOD_DATA_BITS - 1, 0);
                s3data.RANGE(jj, MOD_DATA_BITS) = y.range((MOD_DATA_BITS - 1), 0);
            }
            s3.write(s3data);
        }
    }
}

// -------------------------
// K4: postprocess + store
// out[k] = clamp(s3[k] * gamma + delta, 0, 7.9)
// -------------------------
static void func_k4(hls::stream<stype_t>& s3, data_t out[N]) {
    wide_dt (*out_wide) = reinterpret_cast<wide_dt *>(out);

    // const modified_coef_t gamma = (modified_coef_t)1.25;
    const modified_coef_t delta = (modified_coef_t)0.05;

    for (int k = 0; k < N / FACTOR; k++) {
#pragma HLS PIPELINE II = 1
        stype_t s3data = s3.read();
        wide_dt out_data = (wide_dt)0;
        for (int jj = 0; jj < FACTOR; jj++) {
#pragma HLS UNROLL
            modified_data_t in_data;
            in_data.range((MOD_DATA_BITS - 1), 0) = s3data.RANGE(jj, MOD_DATA_BITS);
            modified_data_t in_q = in_data >> 2;
            modified_data_t sum_q = in_data + in_q;
            modified_data_t z = sum_q + delta;
            // z = clamp_fp(z, (modified_data_t)0, (modified_data_t)7.9);
            data_t unmodified_data = (data_t)z;
            out_data.RANGE(jj, IN_DATA_BITS) = unmodified_data.range(IN_DATA_BITS - 1, 0);
        }
        out_wide[k] = out_data;
    }
}

void top_kernel(const data_t in[N],
                data_t out[N]) {
#pragma HLS interface m_axi port=in offset=slave bundle=gmem0 max_widen_bitwidth=1024
#pragma HLS interface m_axi port=out offset=slave bundle=gmem1 max_widen_bitwidth=1024
#pragma HLS interface s_axilite port=return

#pragma HLS DATAFLOW
    /* FIFOS */
    hls::stream<wide_dt> in_data;
#pragma HLS stream variable=in_data depth=32

    hls::stream<stype_t> s0_1;
    hls::stream<stype_t> s0_2;
    hls::stream<stype_t> s1;          // after transform
    hls::stream<modified_stat_t> stats;         // 1 stat per block
    hls::stream<modified_stat_t> inv_stats;         // 1 stat per block
    hls::stream<stype_t>  s3;         // after normalize
#pragma HLS stream variable=s0_1 depth=32
#pragma HLS stream variable=s0_2 depth=32
#pragma HLS stream variable=s1 depth=512
#pragma HLS stream variable=s3 depth=512
#pragma HLS stream variable=stats depth=128
#pragma HLS stream variable=inv_stats depth=128
#pragma HLS bind_storage variable=s0_1 type=fifo impl=lutram
#pragma HLS bind_storage variable=s0_2 type=fifo impl=lutram
    
    // stage 0
    stream_from_dram(in, in_data);
    //stage 1
    func_k0(in_data, s0_1, s0_2);
    // stage 2
    func_k1(s0_1, s1);
    func_k2(s0_2, stats);

    func_sub(stats, inv_stats);

    // stage 3
    func_k3(inv_stats, s1, s3);
    // stage 4
    func_k4(s3, out);
}
