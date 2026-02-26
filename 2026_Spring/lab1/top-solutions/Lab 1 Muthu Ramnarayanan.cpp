#include "dcl.h"
#include <ap_int.h>

#define CHUNKS_PER_ROW (N_COLS / 16)

// =============================================================================
// PRECISION-OPTIMIZED TYPEDEFS
// =============================================================================
// Row sum accumulator: sum of N_COLS (64) values, each up to ~1024, plus 1.0 bias
// Max value: 64 * 1024 + 1 = 65537 → needs ceil(log2(65537)) = 17 integer bits
// Keep 14 fractional bits to match data_t precision
typedef ap_ufixed<31, 17> row_sum_t;

// Division/multiplication intermediate:
// - Division: 1.0 / row_sum gives result in (2^-17, 1] → needs 1 integer bit
// - Fractional bits: 17 (for smallest divisor) + 14 (input precision) = 31
// - Total: 32 bits (reduced from 40, shortens critical path)
typedef ap_ufixed<32, 1, AP_RND, AP_SAT> wide_t;

// Truncation type for normalized values (matches data_t but with truncation)
typedef ap_ufixed<24, 10, AP_TRN, AP_SAT> data_trn_t;

// Column sum accumulator: sum of N_ROWS (256) normalized values, each in [0,1]
// Max value: 256 → needs 9 integer bits, 14 fractional
typedef ap_ufixed<23, 9> col_sum_t;

// Custom packing types
typedef ap_uint<512> u512_pack;
typedef ap_uint<32>  u32_flat;

// Helper to unpack 512-bit chunk into 16 data values
void unpack_chunk(u512_pack chunk, data_t out[16]) {
    for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
        u32_flat raw = chunk.range((k + 1) * 32 - 1, k * 32);
        out[k].range(23, 0) = raw.range(23, 0);
    }
}

// Helper to pack 16 data values into 512-bit chunk
u512_pack pack_chunk(data_t in[16]) {
    u512_pack chunk = 0;
    for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
        u32_flat raw = 0;
        raw.range(23, 0) = in[k].range(23, 0);
        chunk.range((k + 1) * 32 - 1, k * 32) = raw;
    }
    return chunk;
}

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
    
    #pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A max_widen_bitwidth=512
    #pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=return

    u512_pack* A_ptr = (u512_pack*)A_DRAM;
    u512_pack* C_ptr = (u512_pack*)C_DRAM;

    // ---------------------------------------------------------
    // BUFFER ALLOCATION & PARTITIONING
    // ---------------------------------------------------------
    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=2

    // Column sums: accumulate 256 normalized values (each in [0,1])
    col_sum_t col_sum[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=col_sum complete

    data_t row_buf0[N_COLS];
    data_t row_buf1[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=row_buf0 cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=row_buf1 cyclic factor=16 dim=1

    init_col_sum: for (int j = 0; j < N_COLS; j++) {
        #pragma HLS UNROLL factor=16
        col_sum[j] = (col_sum_t)0;
    }

    // Row sum accumulator: needs 17 integer bits to avoid overflow
    row_sum_t current_row_sum = 0;

    // ---------------------------------------------------------
    // PHASE 1: ROW NORMALIZATION (Pipelined with Ping-Pong)
    // ---------------------------------------------------------

    // --- READ ROW 0 (Fixed for Timing) ---
    // Use partial accumulators to break the adder chain
    row_sum_t partial_init[8];
    #pragma HLS ARRAY_PARTITION variable=partial_init complete
    for(int p=0; p<8; p++) {
        #pragma HLS UNROLL
        partial_init[p] = (row_sum_t)0; 
    }
    
    read_row0: for (int ch = 0; ch < CHUNKS_PER_ROW; ch++) {
        #pragma HLS PIPELINE II=1
        u512_pack chunk = A_ptr[ch];
        
        data_t v[16];
        #pragma HLS ARRAY_PARTITION variable=v complete
        unpack_chunk(chunk, v);

        for(int k=0; k<16; k++) {
            #pragma HLS UNROLL
            row_buf0[ch*16 + k] = v[k];
            // Distribute 16 adds across 8 accumulators (depth = 2)
            partial_init[k % 8] += (row_sum_t)v[k]; 
        }
    }

    // Final reduction + Bias (1.0)
    row_sum_t acc = (row_sum_t)1.0; 
    for(int p=0; p<8; p++) {
        #pragma HLS UNROLL
        acc += partial_init[p];
    }
    current_row_sum = acc;


    // --- MAIN LOOP ---
    main_loop: for (int i = 0; i < N_ROWS; i++) {
    
        bool is_even = (i % 2 == 0);
        
        // 1. Calculate Inverse
        // IMPORTANT: Do NOT cast current_row_sum to wide_t - it would saturate!
        // wide_t has only 1 integer bit, but row_sum can be up to 65537
        wide_t inv_denom = wide_t(1.0) / current_row_sum;
        
        // 2. Prepare Accumulators for Next Row (using wider type to avoid overflow)
        // Initialize 0th bucket with Bias (1.0) for bit-exactness
        row_sum_t partial_acc[8];
        #pragma HLS ARRAY_PARTITION variable=partial_acc complete
        partial_acc[0] = (row_sum_t)1.0;
        for(int p=1; p<8; p++) {
            #pragma HLS UNROLL
            partial_acc[p] = (row_sum_t)0;
        }
    
        process_read_loop: for (int ch = 0; ch < CHUNKS_PER_ROW; ch++) {
            #pragma HLS PIPELINE II=1
            
            // --- A. PROCESS STAGE (Row i) ---
            data_t p_vec[16];
            #pragma HLS ARRAY_PARTITION variable=p_vec complete
            
            int base_idx = ch * 16;
            
            if (is_even) {
                for(int k=0; k<16; k++) p_vec[k] = row_buf0[base_idx + k];
            } else {
                for(int k=0; k<16; k++) p_vec[k] = row_buf1[base_idx + k];
            }
    
            for(int k=0; k<16; k++) {
                #pragma HLS UNROLL
                data_t val = p_vec[k];
                // Don't cast val to wide_t (val can be up to 1024, would saturate)
                // Product val * inv_denom = val / row_sum is in [0,1] mathematically
                wide_t raw_product = val * inv_denom;
                data_trn_t saturated = (data_trn_t)raw_product;
                data_t norm = (data_t)saturated;
                
                tmp[i][base_idx + k] = norm;
                col_sum[base_idx + k] += (col_sum_t)norm; 
            }
    
            // --- B. READ STAGE (Row i+1) ---
            if (i < N_ROWS - 1) {
                u512_pack chunk_in = A_ptr[(i + 1) * CHUNKS_PER_ROW + ch];
                
                data_t r_vec[16];
                #pragma HLS ARRAY_PARTITION variable=r_vec complete
                unpack_chunk(chunk_in, r_vec);
    
                row_sum_t chunk_sum = (row_sum_t)0;
                for(int k=0; k<16; k++) {
                    #pragma HLS UNROLL
                    chunk_sum += (row_sum_t)r_vec[k]; // Tree reduction
                    
                    if (is_even) row_buf1[base_idx + k] = r_vec[k];
                    else         row_buf0[base_idx + k] = r_vec[k];
                }
    
                // Partial accumulation to break dependency
                partial_acc[ch % 8] += chunk_sum;
            }
        } 
    
        // 3. Final Summation
        row_sum_t total_read_sum = (row_sum_t)0;
        for(int p=0; p<8; p++) {
            #pragma HLS UNROLL
            total_read_sum += partial_acc[p];
        }
        current_row_sum = total_read_sum;
    }

    // ---------------------------------------------------------
    // PHASE 2: SCALING & WRITE BACK
    // ---------------------------------------------------------
    // INV_N = 1/256 = 2^-8
    // IMPORTANT: Don't cast N_ROWS to wide_t (256 > 2, would saturate)
    wide_t INV_N = wide_t(1.0) / N_ROWS;
    data_t scale[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=16 dim=1

    calc_scale_loop: for (int j = 0; j < N_COLS; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=16
        // Don't cast col_sum to wide_t (col_sum can be up to 256, would saturate)
        // Product col_sum[j] * INV_N is in [0,1], fits in result type
        scale[j] = (data_t)(data_trn_t)(col_sum[j] * INV_N);
    }

    write_row_loop: for (int r = 0; r < N_ROWS; r++) {
        write_chunk_loop: for (int ch = 0; ch < CHUNKS_PER_ROW; ch++) {
            #pragma HLS PIPELINE II=1
            
            data_t v_out[16];
            #pragma HLS ARRAY_PARTITION variable=v_out complete
            int base_idx = ch * 16;

            for (int k = 0; k < 16; k++) {
                #pragma HLS UNROLL
                int c = base_idx + k;
                data_t t_val = tmp[r][c];
                data_t s_val = scale[c];
                v_out[k] = t_val * s_val;
            }
            C_ptr[r * CHUNKS_PER_ROW + ch] = pack_chunk(v_out);
        }
    }
}