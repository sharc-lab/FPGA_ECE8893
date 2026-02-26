#include "dcl.h"
#include <hls_stream.h>

// 20-bit: 3位整数，17位小数。AP_TRN 截断模式消灭多余进位链
typedef ap_ufixed<20, 3> internal_t;

// 常量，强迫走 DSP48 的 18-bit 端口 B
const ap_ufixed<18, 0> WA = 0.10;

typedef ap_uint<512> bus_t;

// 黄金比例并行度：8像素/周期，既能跑到极高频率又不会撑爆 LUT
struct vec8_int {
    internal_t d[8];
};

// =========================================================================
// I/O: AXI Burst 读取
// =========================================================================
void load_bus(const bus_t* in, hls::stream<vec8_int>& out) {
    for (int i = 0; i < (NX * NY) / 16; i++) {
        bus_t cache = in[i]; 
        for (int chunk = 0; chunk < 2; chunk++) { 
            #pragma HLS PIPELINE II=1
            vec8_int p;
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                ap_uint<32> raw = cache.range((chunk * 8 + k + 1) * 32 - 1, (chunk * 8 + k) * 32);
                data_t temp;
                temp.range() = raw.range(23, 0);
                p.d[k] = (internal_t)temp;
            }
            out.write(p);
        }
    }
}

void store_bus(hls::stream<vec8_int>& in, bus_t* out) {
    for (int i = 0; i < (NX * NY) / 16; i++) {
        bus_t cache = 0;
        for (int chunk = 0; chunk < 2; chunk++) {
            #pragma HLS PIPELINE II=1
            vec8_int p = in.read();
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                data_t temp = (data_t)p.d[k];
                ap_uint<32> raw = 0;
                raw.range(23, 0) = temp.range();
                cache.range((chunk * 8 + k + 1) * 32 - 1, (chunk * 8 + k) * 32) = raw;
            }
        }
        out[i] = cache; 
    }
}

// =========================================================================
// CORE: Streaming Stencil Processing Element
// =========================================================================
template<int ID>
void timestep_pe(hls::stream<vec8_int>& in, hls::stream<vec8_int>& out) {
    vec8_int lb[2][NY / 8];
    #pragma HLS ARRAY_PARTITION variable=lb complete dim=1
    #pragma HLS DEPENDENCE variable=lb type=inter false

    for (int i = 0; i < NX + 1; i++) {
        vec8_int top_prev, mid_prev, bot_prev;
        vec8_int top_curr, mid_curr, bot_curr;

        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            top_prev.d[k] = 0; mid_prev.d[k] = 0; bot_prev.d[k] = 0;
            top_curr.d[k] = 0; mid_curr.d[k] = 0; bot_curr.d[k] = 0;
        }

        for (int j = 0; j < NY / 8 + 1; j++) { 
            #pragma HLS PIPELINE II=1
            
            vec8_int bot_next, mid_next, top_next;
            if (i < NX && j < NY / 8) {
                bot_next = in.read();
            } else {
                for(int k = 0; k < 8; k++) { 
                    #pragma HLS UNROLL 
                    bot_next.d[k] = 0; 
                }
            }

            if (j < NY / 8) {
                top_next = lb[0][j];
                mid_next = lb[1][j];
                if (i < NX) {
                    lb[0][j] = mid_next;
                    lb[1][j] = bot_next;
                }
            } else {
                for(int k = 0; k < 8; k++) { 
                    #pragma HLS UNROLL 
                    top_next.d[k] = 0; mid_next.d[k] = 0; 
                }
            }

            if (i > 0 && j > 0) {
                vec8_int out_val;
                int row = i - 1;
                int col_blk = j - 1;

                internal_t w[3][10];
                #pragma HLS ARRAY_PARTITION variable=w complete dim=0

                w[0][0] = top_prev.d[7]; w[1][0] = mid_prev.d[7]; w[2][0] = bot_prev.d[7];
                for(int c = 0; c < 8; c++) {
                    #pragma HLS UNROLL
                    w[0][c+1] = top_curr.d[c];
                    w[1][c+1] = mid_curr.d[c];
                    w[2][c+1] = bot_curr.d[c];
                }
                w[0][9] = top_next.d[0]; w[1][9] = mid_next.d[0]; w[2][9] = bot_next.d[0];

                internal_t col_sum[10];
                #pragma HLS ARRAY_PARTITION variable=col_sum complete
                for(int c = 0; c < 10; c++) {
                    #pragma HLS UNROLL
                    col_sum[c] = w[0][c] + w[2][c];
                }

                for(int v = 0; v < 8; v++) {
                    #pragma HLS UNROLL
                    int col = col_blk * 8 + v;

                    if (row == 0 || row == NX - 1 || col == 0 || col == NY - 1) {
                        out_val.d[v] = w[1][v+1];
                    } 
                    else {
                        internal_t mid = w[1][v+1];
                        internal_t s_ax = col_sum[v+1] + (w[1][v] + w[1][v+2]);
                        internal_t s_dg = col_sum[v]   + col_sum[v+2];

                        internal_t sum_adj = s_ax + (s_dg * (internal_t)0.25);
                        
                        internal_t mul_res = sum_adj * WA;
                        
                        // ==============================================================
                        // 性能阀门：3 级 DSP 流水线，强制打断组合逻辑，压榨极速时钟！
                        // ==============================================================
                        #pragma HLS BIND_OP variable=mul_res op=mul impl=dsp latency=3
                        
                        internal_t res = (mid * (internal_t)0.50) + mul_res;
                        
                        // 末端再补一刀寄存器，确保没有任何长尾延迟
                        #pragma HLS BIND_OP variable=res op=add impl=fabric latency=1
                        
                        out_val.d[v] = res;
                    }
                }
                out.write(out_val);
            }

            top_prev = top_curr; top_curr = top_next;
            mid_prev = mid_curr; mid_curr = mid_next;
            bot_prev = bot_curr; bot_curr = bot_next;
        }
    }
}

// =========================================================================
// TOP KERNEL
// =========================================================================
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    #pragma HLS interface m_axi port=A_in bundle=gmem0 offset=slave depth=65536
    #pragma HLS interface m_axi port=A_out bundle=gmem1 offset=slave depth=65536
    #pragma HLS interface s_axilite port=return

    hls::stream<vec8_int> s[31];
    #pragma HLS STREAM variable=s depth=128

    #pragma HLS DATAFLOW

    load_bus((const bus_t*)A_in, s[0]);
    
    timestep_pe<0>(s[0], s[1]);   timestep_pe<1>(s[1], s[2]);
    timestep_pe<2>(s[2], s[3]);   timestep_pe<3>(s[3], s[4]);
    timestep_pe<4>(s[4], s[5]);   timestep_pe<5>(s[5], s[6]);
    timestep_pe<6>(s[6], s[7]);   timestep_pe<7>(s[7], s[8]);
    timestep_pe<8>(s[8], s[9]);   timestep_pe<9>(s[9], s[10]);
    timestep_pe<10>(s[10], s[11]); timestep_pe<11>(s[11], s[12]);
    timestep_pe<12>(s[12], s[13]); timestep_pe<13>(s[13], s[14]);
    timestep_pe<14>(s[14], s[15]); timestep_pe<15>(s[15], s[16]);
    timestep_pe<16>(s[16], s[17]); timestep_pe<17>(s[17], s[18]);
    timestep_pe<18>(s[18], s[19]); timestep_pe<19>(s[19], s[20]);
    timestep_pe<20>(s[20], s[21]); timestep_pe<21>(s[21], s[22]);
    timestep_pe<22>(s[22], s[23]); timestep_pe<23>(s[23], s[24]);
    timestep_pe<24>(s[24], s[25]); timestep_pe<25>(s[25], s[26]);
    timestep_pe<26>(s[26], s[27]); timestep_pe<27>(s[27], s[28]);
    timestep_pe<28>(s[28], s[29]); timestep_pe<29>(s[29], s[30]);

    store_bus(s[30], (bus_t*)A_out);
}