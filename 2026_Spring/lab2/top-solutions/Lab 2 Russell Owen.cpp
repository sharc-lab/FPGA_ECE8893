#include "dcl.h"
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_fixed.h>

// -------------------------------------------------------------------------
// CONFIGURATION
// -------------------------------------------------------------------------
#define VEC_SIZE 8
#define GRID_WIDTH NX
#define GRID_HEIGHT NY
#define VECS_PER_ROW (GRID_WIDTH / VEC_SIZE)
#define TOTAL_VECS (VECS_PER_ROW * GRID_HEIGHT)

// --- Data Types ---
// Storage format: 16-bit for memory efficiency
typedef ap_fixed<16, 3, AP_RND_CONV, AP_SAT> small_t;  // 13 fractional bits, range [-4, 4)
typedef ap_uint<16> raw_t;
typedef ap_uint<16 * VEC_SIZE> wide_t;  // 128 bits per vector

// Computation types: Right-sized for DSP and LUT optimization
typedef ap_fixed<18, 0, AP_RND, AP_SAT> weight_t;   // 18 fractional bits, used for 0.10 multiplier

// sum4_t: Max sum of 4 small_t values.
// small_t has 3 integer bits. 4-way sum adds 2 int bits.
// We keep the exact 13 fractional bits from small_t. Total: 18 bits.
typedef ap_fixed<18, 5> sum4_t; 

// diag_shift_t: For the * 0.25 operation.
// * 0.25 pushes data 2 bits to the right. To prevent losing those 2 LSBs,
// we add 2 fractional bits. Total: 19 bits (from 18_5 to 19_5).
typedef ap_fixed<19, 5> diag_shift_t;

// mult_res_t: For the * wa operation.
// 22 bits is plenty to capture the output of the DSP slice multiplier.
typedef ap_fixed<22, 5> mult_res_t;

// -------------------------------------------------------------------------
// HELPER FUNCTIONS
// -------------------------------------------------------------------------
wide_t pack(small_t vals[VEC_SIZE]) {
#pragma HLS INLINE
    wide_t res;
    PACK_LOOP:
    for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
        raw_t bits = vals[k].range(15, 0);
        res.range(k * 16 + 15, k * 16) = bits;
    }
    return res;
}

void unpack(wide_t w, small_t vals[VEC_SIZE]) {
#pragma HLS INLINE
    UNPACK_LOOP:
    for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
        raw_t bits = w.range(k * 16 + 15, k * 16);
        vals[k].range(15, 0) = bits;
    }
}

// -------------------------------------------------------------------------
// COMPUTE KERNEL
// -------------------------------------------------------------------------
void compute_step(hls::stream<wide_t>& stream_in,
                  hls::stream<wide_t>& stream_out)
{
    const weight_t wa = 0.10;

    wide_t lb[2][VECS_PER_ROW];
#pragma HLS BIND_STORAGE variable=lb type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=lb complete dim=1

    // Fully partitioned into flip-flops (dim=0), BIND_STORAGE removed
    wide_t vec_window[3][3];
#pragma HLS ARRAY_PARTITION variable=vec_window complete dim=0

    LOOP_GRID:
    for (int i = 0; i < TOTAL_VECS + VECS_PER_ROW + 1; i++) {
#pragma HLS PIPELINE II=1

        wide_t new_vec;
        if (i < TOTAL_VECS) {
            new_vec = stream_in.read();
        } else {
            new_vec = 0; // Zero-padding drain phase
        }

        int col_idx = i % VECS_PER_ROW;
        wide_t top_vec = lb[0][col_idx];
        wide_t mid_vec = lb[1][col_idx];
        
        lb[0][col_idx] = mid_vec;
        lb[1][col_idx] = new_vec;

        SHIFT_WINDOW:
        for(int r=0; r<3; r++) {
#pragma HLS UNROLL
            vec_window[r][0] = vec_window[r][1];
            vec_window[r][1] = vec_window[r][2];
        }
        vec_window[0][2] = top_vec;
        vec_window[1][2] = mid_vec;
        vec_window[2][2] = new_vec;

        int curr_vec_idx = i - VECS_PER_ROW - 1;

        if (curr_vec_idx >= 0 && curr_vec_idx < TOTAL_VECS) {
            int r = curr_vec_idx / VECS_PER_ROW;
            int c = curr_vec_idx % VECS_PER_ROW;
            small_t out_vals[VEC_SIZE];
#pragma HLS ARRAY_PARTITION variable=out_vals complete

            // 1. LOCAL ROW BUFFERS (The "Mux Killer")
            small_t r0[VEC_SIZE + 2], r1[VEC_SIZE + 2], r2[VEC_SIZE + 2];
#pragma HLS ARRAY_PARTITION variable=r0 complete
#pragma HLS ARRAY_PARTITION variable=r1 complete
#pragma HLS ARRAY_PARTITION variable=r2 complete

            // Left halo
            r0[0].range(15,0) = vec_window[0][0].range((VEC_SIZE-1)*16+15, (VEC_SIZE-1)*16);
            r1[0].range(15,0) = vec_window[1][0].range((VEC_SIZE-1)*16+15, (VEC_SIZE-1)*16);
            r2[0].range(15,0) = vec_window[2][0].range((VEC_SIZE-1)*16+15, (VEC_SIZE-1)*16);

            // Center
            for(int j=0; j<VEC_SIZE; j++) {
#pragma HLS UNROLL
                r0[j+1].range(15,0) = vec_window[0][1].range(j*16+15, j*16);
                r1[j+1].range(15,0) = vec_window[1][1].range(j*16+15, j*16);
                r2[j+1].range(15,0) = vec_window[2][1].range(j*16+15, j*16);
            }

            // Right halo
            r0[VEC_SIZE+1].range(15,0) = vec_window[0][2].range(15, 0);
            r1[VEC_SIZE+1].range(15,0) = vec_window[1][2].range(15, 0);
            r2[VEC_SIZE+1].range(15,0) = vec_window[2][2].range(15, 0);

            bool x_boundary = (r == 0) || (r == NX - 1);
            int base_y = c * VEC_SIZE;

            COMPUTE_PARALLEL:
            for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL 
                
                int glob_y = base_y + k;
                bool is_boundary = x_boundary || (glob_y == 0) || (glob_y == NY - 1);

                if (is_boundary) {
                    out_vals[k] = r1[k+1];
                } else {
                    // Balanced addition
                    sum4_t axis_sum = (sum4_t)((sum4_t)r0[k+1] + (sum4_t)r2[k+1]) + 
                                      (sum4_t)((sum4_t)r1[k]   + (sum4_t)r1[k+2]);
                                      
                    sum4_t diag_sum = (sum4_t)((sum4_t)r0[k]   + (sum4_t)r0[k+2]) + 
                                      (sum4_t)((sum4_t)r2[k]   + (sum4_t)r2[k+2]);

                    // Lossless shift into multiplier
                    diag_shift_t weighted_diag = (diag_shift_t)diag_sum * (diag_shift_t)0.25;
                    diag_shift_t combined_neighbors = (diag_shift_t)axis_sum + weighted_diag;

                    mult_res_t wa_mult = (mult_res_t)combined_neighbors * wa;
                    
                    // Final unbiased assignment
                    out_vals[k] = (small_t)((small_t)r1[k+1] * (small_t)0.5 + wa_mult);
                }
            }
            stream_out.write(pack(out_vals));
        }
    }
}

void compute_all_timesteps(hls::stream<wide_t>& stream_in,
                           hls::stream<wide_t>& stream_out)
{
#pragma HLS DATAFLOW

    hls::stream<wide_t> inter[TSTEPS - 1];
#pragma HLS STREAM variable=inter depth=4

    compute_step(stream_in, inter[0]);

TIME_UNROLL:
    for (int t = 1; t < TSTEPS - 1; t++) {
#pragma HLS UNROLL
        compute_step(inter[t - 1], inter[t]);
    }

    compute_step(inter[TSTEPS - 2], stream_out);
}

// -------------------------------------------------------------------------
// LOAD / STORE MODULES
// -------------------------------------------------------------------------
void load_grid(const data_t A_in[NX][NY],
               hls::stream<wide_t>& stream_out)
{
#pragma HLS INLINE off

    const data_t* flat_ptr = (const data_t*)A_in;
    ap_uint<512>* in_ptr   = (ap_uint<512>*)flat_ptr;

LOAD_OUTER:
    for (int i = 0; i < TOTAL_VECS; i += 2) {  
        // PIPELINE II=2 because writing to the stream twice takes 2 cycles.
#pragma HLS PIPELINE II=2

        ap_uint<512> raw = in_ptr[i / 2];

        // First vector (elements 0-7)
        small_t buffer0[VEC_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer0 complete
        
        for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
            ap_int<24> val_bits = raw.range(k * 32 + 23, k * 32);
            data_t val;
            val.range(23,0) = val_bits;
            buffer0[k] = (small_t)val;
        }
        stream_out.write(pack(buffer0));

        // Second vector (elements 8-15)
        small_t buffer1[VEC_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer1 complete
        
        for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
            ap_int<24> val_bits = raw.range((k + VEC_SIZE) * 32 + 23, (k + VEC_SIZE) * 32);
            data_t val;
            val.range(23,0) = val_bits;
            buffer1[k] = (small_t)val;
        }
        stream_out.write(pack(buffer1));
    }
}

void store_grid(hls::stream<wide_t>& stream_in,
                data_t A_out[NX][NY])
{
#pragma HLS INLINE off

    data_t* flat_ptr = (data_t*)A_out;
    ap_uint<512>* out_ptr = (ap_uint<512>*)flat_ptr;

STORE_OUTER:
    for (int i = 0; i < TOTAL_VECS; i += 2) {  
        // PIPELINE II=2 because reading from the stream twice takes 2 cycles.
#pragma HLS PIPELINE II=2

        ap_uint<512> out_pack = 0;

        // First vector
        small_t vals0[VEC_SIZE];
#pragma HLS ARRAY_PARTITION variable=vals0 complete
        wide_t w0 = stream_in.read();
        unpack(w0, vals0);
        
        for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
            data_t d_val = (data_t)vals0[k];
            out_pack.range(k * 32 + 23, k * 32) = d_val.range(23,0);
        }

        // Second vector
        small_t vals1[VEC_SIZE];
#pragma HLS ARRAY_PARTITION variable=vals1 complete
        wide_t w1 = stream_in.read();
        unpack(w1, vals1);
        
        for (int k = 0; k < VEC_SIZE; k++) {
#pragma HLS UNROLL
            data_t d_val = (data_t)vals1[k];
            out_pack.range((k + VEC_SIZE) * 32 + 23, (k + VEC_SIZE) * 32) = d_val.range(23,0);
        }

        out_ptr[i / 2] = out_pack;
    }
}

// -------------------------------------------------------------------------
// TOP FUNCTION
// -------------------------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS INTERFACE m_axi port=A_in offset=slave bundle=gmem0 max_read_burst_length=256 latency=64
#pragma HLS INTERFACE m_axi port=A_out offset=slave bundle=gmem1 max_write_burst_length=256 latency=64
#pragma HLS INTERFACE s_axilite port=return

    hls::stream<wide_t> stream_loaded("stream_loaded");
    hls::stream<wide_t> stream_computed("stream_computed");
    
#pragma HLS STREAM variable=stream_loaded depth=4
#pragma HLS STREAM variable=stream_computed depth=4

#pragma HLS DATAFLOW
    
    load_grid(A_in, stream_loaded);
    compute_all_timesteps(stream_loaded, stream_computed);
    store_grid(stream_computed, A_out);
}