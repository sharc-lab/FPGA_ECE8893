#include "dcl.h"
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Using 12 bits with 0 int which fits precision need (1%)
// Switch to AP_WRAP to save ciruitry
typedef ap_ufixed<12, 0, AP_RND, AP_WRAP> inner_data_t;

// 15 bits 0 int because the accumulation will never be > 1
typedef ap_ufixed<15, 0, AP_RND, AP_WRAP> inner_acc_t;

typedef ap_uint<512> super_wide_t;

struct inner_vector_t {
    inner_data_t p0, p1, p2, p3, p4, p5, p6, p7;
};

// Stages 0 -> 22: Use DSPs
void stencil_PE_dsp(hls::stream<inner_vector_t>& stream_in, hls::stream<inner_vector_t>& stream_out) {
    inner_data_t line_buf[2][NY];
    // #pragma HLS array_partition variable=line_buf type=complete dim=1
    // // Reshape shrinks from 128-bit wide to 96-bit wide
    // #pragma HLS array_reshape variable=line_buf type=cyclic factor=8 dim=2
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS array_partition variable=line_buf type=complete dim=1
    #pragma HLS array_reshape variable=line_buf type=cyclic factor=8 dim=2
    
    inner_data_t window[3][17]; 
    #pragma HLS array_partition variable=window type=complete dim=0

    const inner_data_t wc = (inner_data_t)0.50;
    const inner_data_t wa = (inner_data_t)0.10;
    const inner_data_t wd = (inner_data_t)0.025;

    const int SPATIAL_DELAY = NY + 8; 
    const int TOTAL_CYCLES = (NX * NY) + SPATIAL_DELAY;

    pe_loop: for (int count = 0; count < TOTAL_CYCLES / 8; count++) {
        #pragma HLS pipeline II=1
        
        inner_vector_t val_in = {0,0,0,0,0,0,0,0};
        if (count < (NX * NY) / 8) val_in = stream_in.read();

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 9; c++) window[r][c] = window[r][c + 8];
        }

        int col_idx = (count * 8) % NY;

        window[0][9] = line_buf[0][col_idx+0]; window[1][9] = line_buf[1][col_idx+0]; window[2][9] = val_in.p0;
        window[0][10]= line_buf[0][col_idx+1]; window[1][10]= line_buf[1][col_idx+1]; window[2][10]= val_in.p1;
        window[0][11]= line_buf[0][col_idx+2]; window[1][11]= line_buf[1][col_idx+2]; window[2][11]= val_in.p2;
        window[0][12]= line_buf[0][col_idx+3]; window[1][12]= line_buf[1][col_idx+3]; window[2][12]= val_in.p3;
        window[0][13]= line_buf[0][col_idx+4]; window[1][13]= line_buf[1][col_idx+4]; window[2][13]= val_in.p4;
        window[0][14]= line_buf[0][col_idx+5]; window[1][14]= line_buf[1][col_idx+5]; window[2][14]= val_in.p5;
        window[0][15]= line_buf[0][col_idx+6]; window[1][15]= line_buf[1][col_idx+6]; window[2][15]= val_in.p6;
        window[0][16]= line_buf[0][col_idx+7]; window[1][16]= line_buf[1][col_idx+7]; window[2][16]= val_in.p7;

        for (int k = 0; k < 8; k++) {
            #pragma HLS unroll
            line_buf[0][col_idx + k] = line_buf[1][col_idx + k];
        }
        line_buf[1][col_idx+0] = val_in.p0; line_buf[1][col_idx+1] = val_in.p1;
        line_buf[1][col_idx+2] = val_in.p2; line_buf[1][col_idx+3] = val_in.p3;
        line_buf[1][col_idx+4] = val_in.p4; line_buf[1][col_idx+5] = val_in.p5;
        line_buf[1][col_idx+6] = val_in.p6; line_buf[1][col_idx+7] = val_in.p7;

        if (count >= SPATIAL_DELAY / 8) {
            int linear_out = count - (SPATIAL_DELAY / 8);
            int row_out = linear_out / (NY / 8);
            int col_out = (linear_out % (NY / 8)) * 8;

            bool is_tb = (row_out == 0 || row_out == NX - 1);
            bool is_left = (col_out == 0);
            bool is_right = ((col_out + 7) == NY - 1);

            inner_vector_t val_out;
            inner_data_t v_out_arr[8];
            #pragma HLS array_partition variable=v_out_arr type=complete

            for (int p = 0; p < 8; p++) {
                #pragma HLS unroll
                bool is_bnd = is_tb || (p == 0 && is_left) || (p == 7 && is_right);
                if (is_bnd) {
                    v_out_arr[p] = window[1][p + 1]; 
                } else {
                    // Force registers between add and DSP to keep clock period low
                    inner_acc_t sum_axis = (inner_acc_t)window[0][p+1] + window[2][p+1] + window[1][p] + window[1][p+2];
                    inner_acc_t sum_axis_reg = sum_axis;
                    inner_acc_t sum_diag = (inner_acc_t)window[0][p] + window[0][p+2] + window[2][p] + window[2][p+2];
                    inner_acc_t sum_diag_reg = sum_diag;
                    
                    // inner_acc_t mult_wa = sum_axis * wa;
                    // inner_acc_t mult_wd = sum_diag * wd;
                    inner_acc_t mult_wa = sum_axis_reg * wa;
                    inner_acc_t mult_wd = sum_diag_reg * wd;

                    // #pragma HLS bind_op variable=mult_wa op=mul impl=dsp
                    // #pragma HLS bind_op variable=mult_wd op=mul impl=dsp
                    // Force split DSP into 3 steps so they dont take 1 long clock cycle
                    #pragma HLS bind_op variable=mult_wa op=mul impl=dsp latency=3
                    #pragma HLS bind_op variable=mult_wd op=mul impl=dsp latency=3
                    
                    v_out_arr[p] = (inner_data_t)((inner_acc_t)window[1][p+1]*wc + mult_wa + mult_wd);
                }
            }
            val_out.p0 = v_out_arr[0]; val_out.p1 = v_out_arr[1]; val_out.p2 = v_out_arr[2]; val_out.p3 = v_out_arr[3];
            val_out.p4 = v_out_arr[4]; val_out.p5 = v_out_arr[5]; val_out.p6 = v_out_arr[6]; val_out.p7 = v_out_arr[7];
            stream_out.write(val_out);
        }
    }
}

// Last few time stamps: use non-DSPs because otherwise itll run out
void stencil_PE_fabric(hls::stream<inner_vector_t>& stream_in, hls::stream<inner_vector_t>& stream_out) {
    inner_data_t line_buf[2][NY];
    // #pragma HLS array_partition variable=line_buf type=complete dim=1
    // #pragma HLS array_reshape variable=line_buf type=cyclic factor=8 dim=2
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS array_partition variable=line_buf type=complete dim=1
    #pragma HLS array_reshape variable=line_buf type=cyclic factor=8 dim=2
    
    inner_data_t window[3][17]; 
    #pragma HLS array_partition variable=window type=complete dim=0

    const inner_data_t wc = (inner_data_t)0.50;
    const inner_data_t wa = (inner_data_t)0.10;
    const inner_data_t wd = (inner_data_t)0.025;

    const int SPATIAL_DELAY = NY + 8; 
    const int TOTAL_CYCLES = (NX * NY) + SPATIAL_DELAY;

    pe_loop: for (int count = 0; count < TOTAL_CYCLES / 8; count++) {
        #pragma HLS pipeline II=1
        
        inner_vector_t val_in = {0,0,0,0,0,0,0,0};
        if (count < (NX * NY) / 8) val_in = stream_in.read();

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 9; c++) window[r][c] = window[r][c + 8];
        }

        int col_idx = (count * 8) % NY;

        window[0][9] = line_buf[0][col_idx+0]; window[1][9] = line_buf[1][col_idx+0]; window[2][9] = val_in.p0;
        window[0][10]= line_buf[0][col_idx+1]; window[1][10]= line_buf[1][col_idx+1]; window[2][10]= val_in.p1;
        window[0][11]= line_buf[0][col_idx+2]; window[1][11]= line_buf[1][col_idx+2]; window[2][11]= val_in.p2;
        window[0][12]= line_buf[0][col_idx+3]; window[1][12]= line_buf[1][col_idx+3]; window[2][12]= val_in.p3;
        window[0][13]= line_buf[0][col_idx+4]; window[1][13]= line_buf[1][col_idx+4]; window[2][13]= val_in.p4;
        window[0][14]= line_buf[0][col_idx+5]; window[1][14]= line_buf[1][col_idx+5]; window[2][14]= val_in.p5;
        window[0][15]= line_buf[0][col_idx+6]; window[1][15]= line_buf[1][col_idx+6]; window[2][15]= val_in.p6;
        window[0][16]= line_buf[0][col_idx+7]; window[1][16]= line_buf[1][col_idx+7]; window[2][16]= val_in.p7;

        for (int k = 0; k < 8; k++) {
            #pragma HLS unroll
            line_buf[0][col_idx + k] = line_buf[1][col_idx + k];
        }
        line_buf[1][col_idx+0] = val_in.p0; line_buf[1][col_idx+1] = val_in.p1;
        line_buf[1][col_idx+2] = val_in.p2; line_buf[1][col_idx+3] = val_in.p3;
        line_buf[1][col_idx+4] = val_in.p4; line_buf[1][col_idx+5] = val_in.p5;
        line_buf[1][col_idx+6] = val_in.p6; line_buf[1][col_idx+7] = val_in.p7;

        if (count >= SPATIAL_DELAY / 8) {
            int linear_out = count - (SPATIAL_DELAY / 8);
            int row_out = linear_out / (NY / 8);
            int col_out = (linear_out % (NY / 8)) * 8;

            bool is_tb = (row_out == 0 || row_out == NX - 1);
            bool is_left = (col_out == 0);
            bool is_right = ((col_out + 7) == NY - 1);

            inner_vector_t val_out;
            inner_data_t v_out_arr[8];
            #pragma HLS array_partition variable=v_out_arr type=complete

            for (int p = 0; p < 8; p++) {
                #pragma HLS unroll
                bool is_bnd = is_tb || (p == 0 && is_left) || (p == 7 && is_right);
                if (is_bnd) {
                    v_out_arr[p] = window[1][p + 1]; 
                } else {
                    inner_acc_t sum_axis = (inner_acc_t)window[0][p+1] + window[2][p+1] + window[1][p] + window[1][p+2];
                    inner_acc_t sum_diag = (inner_acc_t)window[0][p] + window[0][p+2] + window[2][p] + window[2][p+2];

                    // Explicitly force HLS to register the output of these adder trees
                    // Saves clock period time
                    #pragma HLS bind_op variable=sum_axis op=add impl=fabric latency=1
                    #pragma HLS bind_op variable=sum_diag op=add impl=fabric latency=1
                    
                    inner_acc_t mult_wa = sum_axis * wa;
                    inner_acc_t mult_wd = sum_diag * wd;
                    // Trying to force register between long ~12 stage multiplication path
                    #pragma HLS bind_op variable=mult_wa op=mul impl=fabric
                    #pragma HLS bind_op variable=mult_wd op=mul impl=fabric
                    // #pragma HLS bind_op variable=mult_wa op=mul impl=fabric latency = 1
                    // #pragma HLS bind_op variable=mult_wd op=mul impl=fabric latency = 1
                    
                    v_out_arr[p] = (inner_data_t)((inner_acc_t)window[1][p+1]*wc + mult_wa + mult_wd);
                }
            }
            val_out.p0 = v_out_arr[0]; val_out.p1 = v_out_arr[1]; val_out.p2 = v_out_arr[2]; val_out.p3 = v_out_arr[3];
            val_out.p4 = v_out_arr[4]; val_out.p5 = v_out_arr[5]; val_out.p6 = v_out_arr[6]; val_out.p7 = v_out_arr[7];
            stream_out.write(val_out);
        }
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    #pragma HLS interface m_axi port=A_in register offset=slave bundle=A_in
    #pragma HLS interface m_axi port=A_out register offset=slave bundle=A_out
    #pragma HLS interface s_axilite port=return

    super_wide_t *flat_A_in = (super_wide_t*)A_in;
    super_wide_t *flat_A_out = (super_wide_t*)A_out;
    
    const int TOTAL_CHUNKS = (NX * NY) / 16;

    hls::stream<inner_vector_t> pipes[TSTEPS + 1];
    // #pragma HLS stream depth=2 variable=pipes
    #pragma HLS bind_storage variable=pipes type=fifo impl=srl
    
    #pragma HLS dataflow
    
    read_in: for (int chunk_idx = 0; chunk_idx < TOTAL_CHUNKS; chunk_idx++) {
        super_wide_t chunk = flat_A_in[chunk_idx];
        read_in_k: for (int k = 0; k < 2; k++) {
            #pragma HLS pipeline II=1
            inner_vector_t val;
            inner_data_t temp[8];
            #pragma HLS array_partition variable=temp type=complete

            for (int p = 0; p < 8; p++) {
                #pragma HLS unroll
                unsigned int raw = chunk.range(31 + 32*(8*k + p), 32*(8*k + p));
                data_t ext_p = *(data_t*)(&raw);
                temp[p] = (inner_data_t)ext_p; // Downcast to 12-bit for fast compuation
            }
            
            val.p0 = temp[0]; val.p1 = temp[1]; val.p2 = temp[2]; val.p3 = temp[3];
            val.p4 = temp[4]; val.p5 = temp[5]; val.p6 = temp[6]; val.p7 = temp[7];
            pipes[0].write(val);
        }
    }

    compute_dsp_stages: for (int t = 0; t < 22; t++) {
        #pragma HLS unroll
        stencil_PE_dsp(pipes[t], pipes[t + 1]);
    }
    
    compute_fabric_stages: for (int t = 22; t < TSTEPS; t++) {
        #pragma HLS unroll
        stencil_PE_fabric(pipes[t], pipes[t + 1]);
    }

    write_out: for (int chunk_idx = 0; chunk_idx < TOTAL_CHUNKS; chunk_idx++) {
        super_wide_t chunk = 0;
        write_out_k: for (int k = 0; k < 2; k++) {
            #pragma HLS pipeline II=1
            inner_vector_t val = pipes[TSTEPS].read();
            
            inner_data_t temp[8] = {val.p0, val.p1, val.p2, val.p3, val.p4, val.p5, val.p6, val.p7};
            #pragma HLS array_partition variable=temp type=complete

            for (int p = 0; p < 8; p++) {
                #pragma HLS unroll
                data_t ext_p = (data_t)temp[p]; // Upcast to 32-bit to return to expected output
                unsigned int raw = *(unsigned int*)(&ext_p);
                chunk.range(31 + 32*(8*k + p), 32*(8*k + p)) = raw;
            }
        }
        flat_A_out[chunk_idx] = chunk;
    }
}