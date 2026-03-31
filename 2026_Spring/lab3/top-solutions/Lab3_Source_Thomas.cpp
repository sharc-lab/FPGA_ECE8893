#include "dcl.h"
#include <hls_stream.h>

typedef ap_fixed<16, 3, AP_RND, AP_SAT> internal_data_t;   // main signal
typedef ap_fixed<22, 10, AP_RND, AP_SAT> internal_acc_t;   // accumulator for reductions
typedef ap_fixed<14, 2, AP_RND, AP_SAT> internal_stat_t;  // per-block statistic

typedef ap_uint<1024> bus_t;

struct vec32_int {
    internal_data_t d[32];
};

void load_bus(const bus_t* in, hls::stream<vec32_int>& out) {
    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        bus_t cache = in[i];
        vec32_int p;
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            ap_uint<32> raw = cache.range(k * 32 + 31, k * 32);
            data_t temp;
            temp.range() = raw.range(31, 0);
            p.d[k] = (internal_data_t)temp;
        }
        out.write(p);
    }
}

void store_bus(hls::stream<vec32_int>& in, bus_t* out) {
    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        bus_t cache = 0;
        vec32_int p = in.read();
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            data_t temp = p.d[k];
            ap_uint<32> raw = 0;
            raw.range(31, 0) = temp.range();
            cache.range(k * 32 + 31, k * 32) = raw;
        }
        out[i] = cache; 
    }
}

void preprocess(hls::stream<vec32_int>& in, hls::stream<vec32_int>& out1, hls::stream<vec32_int>& out2) {
    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        vec32_int p = in.read();
        vec32_int q;
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            q.d[k] = (internal_data_t)((internal_acc_t) 0.875 * (internal_acc_t)p.d[k] + (internal_acc_t) 0.125);
        }
        out1.write(q);
        out2.write(q);
    }
}

void transform(hls::stream<vec32_int>& in, hls::stream<vec32_int>& out) {
    static internal_data_t reg_last = 0;
    static internal_data_t reg_second_last = 0;

    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        
        vec32_int p = in.read();
        vec32_int q;
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            internal_data_t x0, x1, x2;
            x0 = p.d[k];

            if (k == 0) {
                x1 = reg_last;
                x2 = reg_second_last;
            } else if (k == 1) {
                x1 = p.d[0];
                x2 = reg_last;
            } else {
                x1 = p.d[k-1];
                x2 = p.d[k-2];
            }

            internal_acc_t acc = (internal_acc_t) (0.5) * x0 + (internal_acc_t) (-0.25) * x1 + (internal_acc_t) (0.125) * x2;
            internal_data_t y = (internal_data_t)acc;
            q.d[k] = (y < 0) ? (internal_data_t)(-y) : y;
        }
        reg_second_last = p.d[30];
        reg_last = p.d[31];
        out.write(q);
    }
}

void block_statistic(hls::stream<vec32_int>& in, hls::stream<internal_data_t>& stat) {
    internal_acc_t block_sum = 0;
    ap_uint<4> vector_count = 0;

    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        vec32_int p = in.read();
        internal_acc_t vector_sum = 0;
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            internal_data_t val = p.d[k];
            internal_data_t val_abs = (val < 0) ? (internal_data_t)-val : val;
            vector_sum += (internal_acc_t)val_abs;
        }
        block_sum += vector_sum;
        vector_count++;
        if (vector_count == 8) {
            internal_acc_t avg_abs;
            avg_abs = (internal_stat_t)(block_sum >> 8);
            stat.write(avg_abs + (internal_data_t) 0.5);
            block_sum = 0;
            vector_count = 0;
        }
    }
}

void compute_normalization(hls::stream<vec32_int>& buffer_in, hls::stream<internal_data_t>& stat, hls::stream<vec32_int>& out) {
    vec32_int local_block[8];
    #pragma HLS ARRAY_PARTITION variable=local_block complete dim=0

    for (int b = 0; b < (N / 256); b++) {
        // Read stat once per block
        internal_stat_t st = stat.read();
        internal_stat_t inv_st = (internal_stat_t)((internal_acc_t)1.0 / (internal_acc_t)st);

        // This is the single pipeline loop
        for (int v = 0; v < 8; v++) {
            #pragma HLS PIPELINE II=1
            vec32_int current_vec = buffer_in.read();
            vec32_int norm_vec;
            for (int k = 0; k < 32; k++) {
                #pragma HLS UNROLL
                internal_acc_t prod = (internal_acc_t)current_vec.d[k] * (internal_acc_t)inv_st;
                norm_vec.d[k] = (internal_data_t)(prod);
            }
            out.write(norm_vec);
        }
    }
}

void postprocess(hls::stream<vec32_int>& in, hls::stream<vec32_int>& out) {
    for (int i = 0; i < N / 32; i++) {
        #pragma HLS PIPELINE II=1
        vec32_int p = in.read();
        vec32_int q;
        internal_acc_t temp;
        for (int k = 0; k < 32; k++) {
            #pragma HLS UNROLL
            temp = (internal_acc_t) 1.25 * (internal_acc_t)p.d[k];
            q.d[k] = (internal_data_t)(temp + (internal_acc_t) 0.05);
        }
        out.write(q);
    }
}

void top_kernel(const data_t in[N], data_t out[N]) {
#pragma HLS interface m_axi port=in offset=slave bundle=in
#pragma HLS interface m_axi port=out offset=slave bundle=out
#pragma HLS interface s_axilite port=return

    hls::stream<vec32_int> s[6];
    hls::stream<internal_data_t> stat;
    #pragma HLS STREAM variable=s depth=4
    #pragma HLS STREAM variable=s[3] depth=1024
    #pragma HLS DATAFLOW

    load_bus((const bus_t*)in, s[0]);
    preprocess(s[0], s[1], s[2]); //s[1] == s[2]
    transform(s[1], s[3]);
    block_statistic(s[2], stat);
    compute_normalization(s[3], stat, s[4]);
    postprocess(s[4], s[5]);
    store_bus(s[5], (bus_t*)out);
}
