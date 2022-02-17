//--------------------------------------------------------------------------
// Except minor height and width increment to IN_BUF_HEIGHT
// and IN_BUF_WIDTH respectively, no other dimension should be modified.
//--------------------------------------------------------------------------

#ifndef CONV_H_
#define CONV_H_

#include <iostream>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Compiler Defines
//--------------------------------------------------------------------------
#ifdef HLS_SIM
    #include "config.h"
#endif

//--------------------------------------------------------------------------
// Type Conversions
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16,2> fm_t;
    typedef ap_fixed<16,2> wt_t;
#endif

#define STRIDE             1
#define PADDING            1 

#define IN_BUF_DEPTH       8
#define IN_BUF_HEIGHT     23 // Hint: 23 + ?? (how many border features are needed?) 
#define IN_BUF_WIDTH      20 // Hint: 20 + ?? (how many border features are needed?)

#define OUT_BUF_DEPTH     16
#define OUT_BUF_HEIGHT    23
#define OUT_BUF_WIDTH     20

#define N_TILE_ROWS (int) (184/OUT_BUF_HEIGHT)
#define N_TILE_COLS (int) (320/OUT_BUF_WIDTH)

//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM (
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[64][184][320], 
    int  ti, 
    int  tj, 
    int  d
);

void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[64][64][3][3],
    wt_t bias[64],
    int b,
    int d
);

void conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3]
);

void save_partial_output_tile_block (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    int  d
);

void store_output_tile_to_DRAM (
    fm_t out_fm[64][184][320], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b
);

void model_conv (
    fm_t input_feature_map[64][184][320],
    wt_t layer_weights[64][64][3][3],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][184][320]
);

void tiled_conv (
    fm_t input_feature_map[64][184][320],
    wt_t layer_weights[64][64][3][3],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][184][320]
);

#endif
