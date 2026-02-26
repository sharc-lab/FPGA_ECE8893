#include "dcl.h"

data_t division_correct(data_t numerator, ap_uint<35> inv_denom,
                        ap_uint<24> denom) {
#pragma hls inline
  ap_uint<24> num_int = (ap_uint<24>)(numerator.range(23, 0));
  ap_uint<64> wide_product = num_int * inv_denom;

  ap_uint<24> q = (ap_uint<24>)(wide_product >> 35);
  ap_uint<17> q_short = (ap_uint<17>)q;

  ap_uint<64> check_mul = (ap_uint<64>)q_short * (ap_uint<64>)denom;
  ap_uint<64> num_int_shifted = (ap_uint<64>)num_int << 14;
  ap_uint<64> rem = num_int_shifted - check_mul;

  if (rem >= denom)
    q++;

  data_t result;
  result.range(23, 0) = q;
  return result;
}

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS], data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port = A_DRAM offset = slave bundle = A
#pragma HLS interface m_axi port = C_DRAM offset = slave bundle = C
#pragma HLS interface s_axilite port = return

  static data_t tmp[N_ROWS][N_COLS];
#pragma HLS ARRAY_PARTITION variable = tmp dim = 1 factor = 2 type = cyclic
#pragma HLS ARRAY_PARTITION variable = tmp dim = 2 type = complete
#pragma HLS BIND_STORAGE variable = tmp type = ram_2p impl = bram

  // Intermediate buffer for column sums
  static data_t tmp_col_sum[N_COLS];
#pragma HLS ARRAY_PARTITION variable = tmp_col_sum dim = 1 type = complete
#pragma HLS BIND_STORAGE variable = tmp_col_sum type = ram_2p impl = bram

  ap_uint<512> *A_WIDE_ACCESS = (ap_uint<512> *)A_DRAM;
  ap_uint<512> *C_WIDE_ACCESS = (ap_uint<512> *)C_DRAM;

init_col_sum_loop:
  for (int j = 0; j < N_COLS; j++) {
#pragma HLS unroll
    tmp_col_sum[j] = 0;
  }

merged_read_norm_loop:
  for (int i = 0; i < N_ROWS; i++) {

    data_t row_sum = 0;
    data_t local_row[N_COLS];
#pragma HLS ARRAY_PARTITION variable = local_row complete

    for (int j = 0; j < N_COLS; j += 16) {
#pragma HLS pipeline II = 1

      ap_uint<512> wide_access = A_WIDE_ACCESS[(i * N_COLS + j) / 16];

      for (int k = 0; k < 16; k++) {
#pragma HLS unroll
        unsigned int raw_bits = wide_access.range(32 * (k + 1) - 1, 32 * k);
        data_t val = *(data_t *)(&raw_bits);

        local_row[j + k] = val;
      }
      data_t chunk_sum = 0;
      for (int k = 0; k < 16; k++) {
#pragma HLS unroll
        chunk_sum += local_row[j + k];
      }
      row_sum += chunk_sum;

      if (j == 48) {
        ap_uint<24> denom = (ap_uint<24>)((row_sum + (data_t)1.0).range(23, 0));
        ap_uint<64> one_shifted = (ap_uint<64>)1 << (35 + 14);
        ap_uint<35> inv_denom = (ap_uint<35>)(one_shifted / denom);

        for (int c = 0; c < N_COLS; c++) {
#pragma HLS unroll
          data_t norm = division_correct(local_row[c], inv_denom, denom);
          tmp[i][c] = norm;
          tmp_col_sum[c] += norm;
        }
      }
    }
  }

merged_scale_write_loop:
  for (int i = 0; i < N_ROWS; i++) {
    for (int j = 0; j < N_COLS; j += 16) {
#pragma HLS pipeline II = 1

      ap_uint<512> wide_pack;

      for (int k = 0; k < 16; k++) {
#pragma HLS unroll

        data_t val = tmp[i][j + k];
        data_t col_sum = tmp_col_sum[j + k];

        data_t res = val * (col_sum / 256);

        unsigned int raw_bits = *(unsigned int *)(&res);
        wide_pack.range(32 * (k + 1) - 1, 32 * k) = raw_bits;
      }

      C_WIDE_ACCESS[(i * N_COLS + j) / 16] = wide_pack;
    }
  }
}