#include "dcl.h"
#include "hls_stream.h"
#include <string>

using namespace std;

#define N_OVER_BLOCK 256
#define VEC_DATA_SIZE 32
#define WIDE_SIZE 1024

typedef ap_uint<WIDE_SIZE> wide_t;
typedef ap_fixed<12, 1, AP_RND, AP_SAT> small_data_t; // New internal type
typedef ap_fixed<27, 9, AP_RND, AP_SAT> small_acc_t; 
typedef ap_fixed<8, 2, AP_RND, AP_SAT> small_stat_t;  // per-block statistic
typedef ap_fixed<3, 0, AP_RND, AP_SAT> small_coef_t;  // coefficients

// Internal struct now uses the smaller type
struct vec_data_t {
    small_data_t val[VEC_DATA_SIZE];
};

static inline small_data_t abs_fp(small_data_t x) {
    #pragma HLS inline
    return (x < (small_data_t)0) ? (small_data_t)(-x) : x;
}

static inline small_data_t clamp_fp_small(small_data_t x, small_data_t lo, small_data_t hi) {
    #pragma HLS inline
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void preprocess(const wide_t* A_wide, hls::stream<vec_data_t>& out_fifo) {
    #pragma HLS inline off
    const small_coef_t beta = (small_coef_t)0.125;

    load_cur: for (int idx = 0; idx < (N) / VEC_DATA_SIZE; idx++) {
        #pragma HLS pipeline II=1
        wide_t word = A_wide[idx];
        vec_data_t vec;
        load_cur_in: for (int k = 0; k < VEC_DATA_SIZE; k++) {
            #pragma HLS unroll
            // Extract 32 bits from wide word, but cast to small_data_t
            data_t raw_val;
            raw_val.range(31, 0) = word.range(k * 32 + 31, k * 32);
            
            small_data_t tmp = (small_data_t)raw_val;
            vec.val[k] = (small_data_t)((small_acc_t)tmp - ((small_acc_t)tmp >> 3) + (small_acc_t)beta);
        }
        out_fifo.write(vec);
    }
}

void transform_k1_k2(hls::stream<vec_data_t>& s0_stream, hls::stream<vec_data_t>& s1_stream, hls::stream<small_stat_t>& stats_stream) {
    small_data_t prev1 = 0;
    small_data_t prev2 = 0;
    const small_stat_t eps = (small_stat_t)0.5;
    const small_acc_t one_over_block = (small_acc_t)1/(small_acc_t)BLOCK;
    
    transform_k1_k2_1: for (int b = 0; b < N_OVER_BLOCK; b++) {
        small_acc_t sum_abs = 0;
        transform_k1_k2_2: for (int i = 0; i < BLOCK/VEC_DATA_SIZE; i++) {
            #pragma HLS pipeline II=1
            vec_data_t curr_vec = s0_stream.read();
            vec_data_t vec_write;
            
            // 1. Manually facilitate an Adder Tree for the 32 elements
            small_acc_t level1[16];
            #pragma HLS array_partition variable=level1 complete
            for(int k=0; k<16; k++) {
                #pragma HLS unroll
                level1[k] = (small_acc_t)abs_fp(curr_vec.val[k*2]) + (small_acc_t)abs_fp(curr_vec.val[k*2+1]);
            }

            small_acc_t level2[8];
            #pragma HLS array_partition variable=level2 complete
            for(int k=0; k<8; k++) {
                #pragma HLS unroll
                level2[k] = level1[k*2] + level1[k*2+1];
            }

            small_acc_t level3[4];
            #pragma HLS array_partition variable=level3 complete
            for(int k=0; k<4; k++) {
                #pragma HLS unroll
                level3[k] = level2[k*2] + level2[k*2+1];
            }

            small_acc_t row_sum = (level3[0] + level3[1]) + (level3[2] + level3[3]);

            // 2. Original Transformation Logic
            transform_k1_k2_3: for(int k = 0; k < VEC_DATA_SIZE; k++) {
                #pragma HLS unroll
                small_data_t x0 = curr_vec.val[k];
                small_data_t x1 = (k == 0) ? prev1 : curr_vec.val[k-1];
                small_data_t x2 = (k == 0) ? prev2 : (k == 1) ? prev1 : curr_vec.val[k-2];
                
                small_acc_t acc = ((small_acc_t)x0 * (small_acc_t)4.0 - (small_acc_t)x1 * (small_acc_t)2.0 + (small_acc_t)x2) >> 3;
                small_data_t y = (small_data_t)acc;
                y = abs_fp(y);
                y = clamp_fp_small(y, (small_data_t)0, (small_data_t)7.5);
                
                vec_write.val[k] = y;
            }

            sum_abs += row_sum;
            prev2 = curr_vec.val[VEC_DATA_SIZE-2];
            prev1 = curr_vec.val[VEC_DATA_SIZE-1];
            s1_stream.write(vec_write);
        }
        
        // 3. Register the average calculation to break the timing path
        small_acc_t final_sum = sum_abs;
        small_stat_t avg_abs = (small_stat_t)(final_sum * one_over_block);
        
        // Ensure this stays outside the inner loop to prevent it being part of the II=1 bottleneck
        small_stat_t result = (small_stat_t)1 / (avg_abs + eps);
        stats_stream.write(result);
    }
}

void join_norm_k3(hls::stream<vec_data_t>& s1_stream, hls::stream<small_stat_t>& stats_stream, hls::stream<vec_data_t>& s3_stream) {
    join_norm_k3_1: for (int b = 0; b < (N_OVER_BLOCK); b++) {
        small_stat_t st = stats_stream.read();
        small_stat_t inv_st = (small_stat_t)(st);

        join_norm_k3_2: for (int i = 0; i < BLOCK/VEC_DATA_SIZE; i++) {
            #pragma HLS pipeline II=1
            vec_data_t curr_vec = s1_stream.read();
            vec_data_t vec_write;
            join_norm_k3_3: for(int k = 0; k < VEC_DATA_SIZE; k++) {
                #pragma HLS unroll
                small_data_t tmp = curr_vec.val[k];
                vec_write.val[k] = (small_data_t)((small_acc_t)tmp * (small_acc_t)inv_st);
            }
            s3_stream.write(vec_write);
        }
    }
}

void store_k4(hls::stream<vec_data_t>& s3_stream, wide_t* out_wide) {
    const small_acc_t delta = (small_acc_t)0.05;
    store_k4_1: for (int i = 0; i < N/VEC_DATA_SIZE; i++) {
        #pragma HLS pipeline II=1
        vec_data_t curr_vec = s3_stream.read();
        wide_t word = 0;
        store_k4_2: for(int k = 0; k < VEC_DATA_SIZE; k++) {
            #pragma HLS unroll
            small_acc_t val = (small_acc_t)curr_vec.val[k];
            small_acc_t scaled = val + (val>>2) + (small_acc_t)delta;
            
            // Clamp and convert back to 32-bit interface type for storage
            small_data_t clamped = clamp_fp_small((small_data_t)scaled, (small_data_t)0, (small_data_t)7.9);
            data_t out_val = (data_t)clamped; 
            word.range(k * 32 + 31, k * 32) = out_val.range(31, 0);
        }
        out_wide[i] = word;
    }
}

// Top Kernel parameters remain unchanged as requested
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=out offset=slave bundle=out max_widen_bitwidth=WIDE_SIZE
    #pragma HLS interface s_axilite port=return
    #pragma HLS interface m_axi port=in offset=slave bundle=in max_widen_bitwidth=WIDE_SIZE

    hls::stream<vec_data_t> s0_stream;
    wide_t* in_wide = (wide_t*) in;
    wide_t* out_wide = (wide_t*) out;

    hls::stream<vec_data_t> s1_stream;
    hls::stream<small_stat_t> stats_stream;
    hls::stream<vec_data_t> s3_stream;

    #pragma HLS aggregate variable=s0_stream compact=bit
    #pragma HLS aggregate variable=s1_stream compact=bit
    #pragma HLS aggregate variable=s3_stream compact=bit

    #pragma HLS stream depth=32 variable=s0_stream
    #pragma HLS stream depth=WIDE_SIZE variable=s1_stream 
    #pragma HLS stream depth=16 variable=s3_stream
    #pragma HLS stream depth=4  variable=stats_stream

    #pragma HLS dataflow
    preprocess(in_wide, s0_stream); 
    transform_k1_k2(s0_stream, s1_stream, stats_stream);
    join_norm_k3(s1_stream, stats_stream, s3_stream);
    store_k4(s3_stream, out_wide);
}