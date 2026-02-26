/*
Author: Pulkit Gupta
- Recognized that multiplication by reciprocal can optimize the division away
    - had to shift the reciprocal to deal with precision issues
- manually fixing the add reduction to take multiple cycles
- high fanout has problems so had to insert manual copies
    - had to add dummy logic for the copy to do an add so it doesnt get synthesized away
Implementation date: Feb 3rd, 2026 6pm
*/

#include "dcl.h"

// 2241 cycles @ 7.177 = 16,083.657 ns
/*
================================================================
== Place & Route Fail Fast
================================================================
+-----------------------------------------------------------+-----------+---------+--------+
| Criteria                                                  | Guideline | Actual  | Status |
+-----------------------------------------------------------+-----------+---------+--------+
| LUT                                                       | 70%       | 60.57%  | OK     |
| FD                                                        | 50%       | 18.23%  | OK     |
| LUTRAM+SRL                                                | 25%       | 48.72%  | REVIEW |
| CARRY8                                                    | 25%       | 34.35%  | REVIEW |
| MUXF7                                                     | 15%       | 10.88%  | OK     |
| LUT Combining                                             | 20%       | 10.38%  | OK     |
| DSP                                                       | 80%       | 100.00% | REVIEW |
| RAMB/FIFO                                                 | 80%       | 29.63%  | OK     |
| DSP+RAMB+URAM (Avg)                                       | 70%       | 64.81%  | OK     |
| BUFGCE* + BUFGCTRL                                        | 24        | 0       | OK     |
| DONT_TOUCH (cells/nets)                                   | 0         | 0       | OK     |
| MARK_DEBUG (nets)                                         | 0         | 0       | OK     |
| Control Sets                                              | 1323      | 79      | OK     |
| Average Fanout for modules > 100k cells                   | 4         | 1.31    | OK     |
| Max Average Fanout for modules > 100k cells               | 4         | 0       | OK     |
| Non-FD high fanout nets > 10k loads                       | 0         | 6       | REVIEW |
+-----------------------------------------------------------+-----------+---------+--------+
| TIMING-6 (No common primary clock between related clocks) | 0         | 0       | OK     |
| TIMING-7 (No common node between related clocks)          | 0         | 0       | OK     |
| TIMING-8 (No common period between related clocks)        | 0         | 0       | OK     |
| TIMING-14 (LUT on the clock tree)                         | 0         | 0       | OK     |
| TIMING-35 (No common node in paths with the same clock)   | 0         | 0       | OK     |
+-----------------------------------------------------------+-----------+---------+--------+
| Number of paths above max LUT budgeting (0.350ns)         | 0         | 0       | OK     |
| Number of paths above max Net budgeting (0.239ns)         | 0         | 0       | OK     |
+-----------------------------------------------------------+-----------+---------+--------+
#=== Post-Implementation Resource usage ===
SLICE:            0
LUT:          42739
FF:           25722
DSP:            360
BRAM:           128
URAM:             0
LATCH:            0
SRL:           6352
CLB:           7954

#=== Final timing ===
CP required:                     10.000
CP achieved post-synthesis:      3.085
CP achieved post-implementation: 7.177
Timing met
*/
#include <ap_fixed.h>
// 16 24 bit values
typedef ap_int<512> packed_t;
// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port=A_DRAM register offset=slave bundle=A 
// max_read_burst_length=256 depth=256 max_widen_bitwidth=512 num_read_outstanding=256
#pragma HLS interface m_axi port=C_DRAM register offset=slave bundle=C 
// max_write_burst_length=256 depth=16 max_widen_bitwidth=512 num_write_outstanding=256
#pragma HLS interface s_axilite port=return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t A_2[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    #pragma HLS bind_op variable=C op=mul impl=dsp latency=4
    #pragma HLS bind_op variable=C_DRAM op=mul impl=dsp latency=4



    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
    static data_t colsums[N_COLS];

    #pragma HLS array_partition variable=colsums type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=A type=cyclic factor=64 dim=2
    #pragma HLS array_partition variable=A_2 type=cyclic factor=64 dim=2
    #pragma HLS array_partition variable=C type=cyclic factor=64 dim=2
    #pragma HLS array_partition variable=tmp type=cyclic factor=64 dim=2
    #pragma HLS array_partition variable=tmp type=cyclic factor=2 dim=1
    
    packed_t *A_packed = reinterpret_cast<packed_t *>(A_DRAM);
    for (int r = 0; r < N_ROWS; r++) {
        for (int p = 0; p < 4; p++) {
            #pragma HLS PIPELINE II=1
            packed_t w = A_packed[r * 4 + p];
            // memcpy(&w, &A_DRAM[r][p*16], sizeof(packed_t));

            for (int k = 0; k < 16; k++) {
                #pragma HLS UNROLL
                ap_int<24> v = w.range(32*k + 23, 32*k);
                A[r][p * 16 + k].range(23,0) = v.range(23,0);
                A_2[r][p * 16 + k].range(23,0) = v.range(23,0);
            }
        }
    }

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i+=2) {
        #pragma HLS pipeline II=1
        data_t row_sum = 0.0;
        data_t A_row[N_COLS];
        data_t row_sum_2 = 0.0;
        data_t A_row_2[N_COLS];
        #pragma HLS bind_op variable=A_row op=mul impl=dsp latency=4
        #pragma HLS array_partition variable=A_row type=cyclic factor=64 dim=1
        #pragma HLS bind_op variable=A_row_2 op=mul impl=dsp latency=4
        #pragma HLS array_partition variable=A_row_2 type=cyclic factor=64 dim=1
        // Compute row sum
        for (int j = 0; j < N_COLS; j+=16) {
            #pragma HLS unroll
            // row_sum += A[i][j];
            data_t a0  = A[i][j + 0];
            data_t a1  = A[i][j + 1];
            data_t a2  = A[i][j + 2];
            data_t a3  = A[i][j + 3];
            data_t a4  = A[i][j + 4];
            data_t a5  = A[i][j + 5];
            data_t a6  = A[i][j + 6];
            data_t a7  = A[i][j + 7];
            data_t a8  = A[i][j + 8];
            data_t a9  = A[i][j + 9];
            data_t a10 = A[i][j + 10];
            data_t a11 = A[i][j + 11];
            data_t a12 = A[i][j + 12];
            data_t a13 = A[i][j + 13];
            data_t a14 = A[i][j + 14];
            data_t a15 = A[i][j + 15];
            A_row[j+0 ] = a0;
            A_row[j+1 ] = a1;
            A_row[j+2 ] = a2;
            A_row[j+3 ] = a3;
            A_row[j+4 ] = a4;
            A_row[j+5 ] = a5;
            A_row[j+6 ] = a6;
            A_row[j+7 ] = a7;
            A_row[j+8 ] = a8;
            A_row[j+9 ] = a9;
            A_row[j+10] = a10;
            A_row[j+11] = a11;
            A_row[j+12] = a12;
            A_row[j+13] = a13;
            A_row[j+14] = a14;
            A_row[j+15] = a15;
            data_t t1 = (a0 + a1);
            data_t t2 = (a2 + a3);
            data_t t3 = (a4 + a5);
            data_t t4 = (a6 + a7);
            data_t t5 = (a8 + a9);
            data_t t6 = (a10 + a11);
            data_t t7 = (a12 + a13);
            data_t t8 = (a14 + a15);
            #pragma HLS bind_op variable=t1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t2 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t3 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t4 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t5 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t6 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t7 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t8 op=add impl=fabric latency=1
            data_t l1, l2, l3, l4, m1, m2;
            #pragma HLS bind_op variable=l1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l2 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l3 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l4 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=m1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=m2 op=add impl=fabric latency=1
            l1 = t1 + t2;
            l2 = t3 + t4;
            l3 = t5 + t6;
            l4 = t7 + t8;
            m1 = l1 + l2;
            m2 = l3 + l4;
            data_t ls;
            #pragma HLS bind_op variable=ls op=add impl=fabric latency=1
            ls = m1 + m2;
            row_sum += ls;
        }

        for (int j = 0; j < N_COLS; j+=16) {
            #pragma HLS unroll
            // row_sum += A[i][j];
            data_t a0  = A_2[i+1][j + 0];
            data_t a1  = A_2[i+1][j + 1];
            data_t a2  = A_2[i+1][j + 2];
            data_t a3  = A_2[i+1][j + 3];
            data_t a4  = A_2[i+1][j + 4];
            data_t a5  = A_2[i+1][j + 5];
            data_t a6  = A_2[i+1][j + 6];
            data_t a7  = A_2[i+1][j + 7];
            data_t a8  = A_2[i+1][j + 8];
            data_t a9  = A_2[i+1][j + 9];
            data_t a10 = A_2[i+1][j + 10];
            data_t a11 = A_2[i+1][j + 11];
            data_t a12 = A_2[i+1][j + 12];
            data_t a13 = A_2[i+1][j + 13];
            data_t a14 = A_2[i+1][j + 14];
            data_t a15 = A_2[i+1][j + 15];
            A_row_2[j+0 ] = a0;
            A_row_2[j+1 ] = a1;
            A_row_2[j+2 ] = a2;
            A_row_2[j+3 ] = a3;
            A_row_2[j+4 ] = a4;
            A_row_2[j+5 ] = a5;
            A_row_2[j+6 ] = a6;
            A_row_2[j+7 ] = a7;
            A_row_2[j+8 ] = a8;
            A_row_2[j+9 ] = a9;
            A_row_2[j+10] = a10;
            A_row_2[j+11] = a11;
            A_row_2[j+12] = a12;
            A_row_2[j+13] = a13;
            A_row_2[j+14] = a14;
            A_row_2[j+15] = a15;
            data_t t1 = (a0 + a1);
            data_t t2 = (a2 + a3);
            data_t t3 = (a4 + a5);
            data_t t4 = (a6 + a7);
            data_t t5 = (a8 + a9);
            data_t t6 = (a10 + a11);
            data_t t7 = (a12 + a13);
            data_t t8 = (a14 + a15);
            #pragma HLS bind_op variable=t1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t2 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t3 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t4 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t5 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t6 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t7 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=t8 op=add impl=fabric latency=1
            data_t l1, l2, l3, l4, m1, m2;
            #pragma HLS bind_op variable=l1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l2 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l3 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=l4 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=m1 op=add impl=fabric latency=1
            #pragma HLS bind_op variable=m2 op=add impl=fabric latency=1
            l1 = t1 + t2;
            l2 = t3 + t4;
            l3 = t5 + t6;
            l4 = t7 + t8;
            m1 = l1 + l2;
            m2 = l3 + l4;
            data_t ls;
            #pragma HLS bind_op variable=ls op=add impl=fabric latency=1
            ls = m1 + m2;
            row_sum_2 += ls;
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;
        data_t denom_2 = row_sum_2 + (data_t)1.0;
        // smallest possible denominator is 0.1, therefore max value is 1/1.1 = 0.9
        // largest possible value is 640, therefore min value is 1/641 = .00156...
        // in order to preserve precision of the quotient, where max A value is encountered,
        // do 512/denom, and then divide product by 512. this should be cheap because of pow2 division
        data_t quot = ((data_t) 512.0) / ((data_t) denom);
        data_t quot_2 = ((data_t) 512.0) / ((data_t) denom_2);
        #pragma HLS bind_op variable=quot op=mul impl=dsp latency=4
        #pragma HLS bind_op variable=quot op=add impl=fabric latency=2
        #pragma HLS bind_op variable=quot_2 op=mul impl=dsp latency=4
        #pragma HLS bind_op variable=quot_2 op=add impl=fabric latency=2
        data_t quot_arr[8];
        data_t quot_arr_2[8];
        #pragma HLS array_partition variable=quot_arr type=cyclic factor=8 dim=1
        #pragma HLS bind_op variable=quot_arr op=add impl=fabric latency=2
        #pragma HLS array_partition variable=quot_arr_2 type=cyclic factor=8 dim=1
        #pragma HLS bind_op variable=quot_arr_2 op=add impl=fabric latency=2
        {
            #pragma HLS latency min=2
            quot_arr[0] = quot + ((data_t) 0);
            quot_arr[1] = quot + ((data_t) 1);
            quot_arr[2] = quot + ((data_t) 2);
            quot_arr[3] = quot + ((data_t) 3);
            quot_arr[4] = quot + ((data_t) 4);
            quot_arr[5] = quot + ((data_t) 5);
            quot_arr[6] = quot + ((data_t) 6);
            quot_arr[7] = quot + ((data_t) 7);
            quot_arr_2[0] = quot_2 + ((data_t) 0);
            quot_arr_2[1] = quot_2 + ((data_t) 1);
            quot_arr_2[2] = quot_2 + ((data_t) 2);
            quot_arr_2[3] = quot_2 + ((data_t) 3);
            quot_arr_2[4] = quot_2 + ((data_t) 4);
            quot_arr_2[5] = quot_2 + ((data_t) 5);
            quot_arr_2[6] = quot_2 + ((data_t) 6);
            quot_arr_2[7] = quot_2 + ((data_t) 7);
        }

        // Normalize each element in the row
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS unroll
            data_t scaled;
            data_t scaled_2;
            #pragma HLS bind_op variable=scaled op=mul impl=dsp latency=4
            #pragma HLS bind_op variable=scaled_2 op=mul impl=dsp latency=4
            #pragma HLS bind_op variable=j op=add impl=fabric latency=2
            scaled = ((data_t)(A_row[j] * (quot_arr[j&7] - ((data_t) (j&7)))));
            scaled_2 = ((data_t)(A_row_2[j] * (quot_arr_2[j&7] - ((data_t) (j&7)))));
            data_t normalized;
            data_t normalized_2;
            normalized = scaled >> 9;
            normalized_2 = scaled_2 >> 9;
            tmp[i][j] = normalized;
            tmp[i+1][j] = normalized_2;
            data_t normsum = normalized + normalized_2;
            colsums[j] += normsum;
        }
    }

    data_t scales[N_COLS];
    #pragma HLS bind_op variable=scales op=mul impl=dsp latency=4
    #pragma HLS bind_op variable=tmp op=mul impl=dsp latency=4

    #pragma HLS array_partition variable=scales type=cyclic factor=64 dim=1
    for (int j = 0; j < N_COLS; j++) {
        // #pragma HLS pipeline II=1
        #pragma HLS unroll
        scales[j] = colsums[j] >> 8; // / (data_t) N_ROWS;
    }
    
    packed_t *C_packed = reinterpret_cast<packed_t *>(C_DRAM);
    for (int i = 0; i < N_ROWS; i++) {
        data_t c_arr[N_COLS];
        #pragma HLS array_partition variable=c_arr type=cyclic factor=64 dim=1
        #pragma HLS bind_op variable=scales op=mul impl=dsp latency=4
        #pragma HLS bind_op variable=tmp op=mul impl=dsp latency=4
        #pragma HLS bind_op variable=c_arr op=mul impl=dsp latency=4
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS unroll
            c_arr[j] = tmp[i][j] * scales[j];
        }
        for (int p = 0; p < 4; p++) {
            #pragma HLS pipeline II=1
            packed_t w = 0;
            {
                #pragma HLS latency min=2
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    // int j = p * 16 + k;
                    // data_t value;
                    // #pragma HLS bind_op variable=value op=mul impl=dsp latency=4
                    // value = tmp[i][j] * scales[j];
                    ap_int<32> lane = 0;
                    lane.range(23,0) = c_arr[p * 16 + k].range(23,0);
                    // lane.range(23,0) = value.range(23,0);
                    w.range(32*k+31,32*k) = lane.range(31,0);
                }
            }
            C_packed[i * 4 + p] = w;
        }
    }
}