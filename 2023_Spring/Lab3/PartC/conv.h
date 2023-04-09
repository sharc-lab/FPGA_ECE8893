///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv.h
// Description: Header file for tiling-based synthesizable implementation of 
//              ResNet-50's first 7x7 convolution layer with HD input image.
///////////////////////////////////////////////////////////////////////////////

#ifndef CONV_H_
#define CONV_H_

#include <iostream>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Type conversions for simulation and synthesis
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16,3> fm_t;
    typedef ap_fixed<16,3> wt_t;
#endif

//--------------------------------------------------------------------------
// Configuration of ResNet-50's first 7 x 7 convolution layer with HD input 
//--------------------------------------------------------------------------
#define IN_FM_DEPTH       3
#define IN_FM_HEIGHT      736
#define IN_FM_WIDTH       1280

#define OUT_FM_DEPTH      64
#define OUT_FM_HEIGHT     368
#define OUT_FM_WIDTH      640

#define STRIDE            2
#define PADDING           3 
#define KERNEL_HEIGHT     7
#define KERNEL_WIDTH      7

//--------------------------------------------------------------------------
// Divide the input image into multiple tiles 
//--------------------------------------------------------------------------
#define TILE_HEIGHT       46
#define TILE_WIDTH        40

#define N_TILE_ROWS (int) IN_FM_HEIGHT / TILE_HEIGHT
#define N_TILE_COLS (int) IN_FM_WIDTH  / TILE_WIDTH

//--------------------------------------------------------------------------
// 3 zero padding rows on top and to the left,
// 2 zero padding rows at bottom and to the right.
// Why is this correct?
//--------------------------------------------------------------------------
#define MARGIN            5

//--------------------------------------------------------------------------
// Input tile buffer dimensions 
//--------------------------------------------------------------------------
#define IN_BUF_DEPTH      3
#define IN_BUF_HEIGHT     TILE_HEIGHT + MARGIN 
#define IN_BUF_WIDTH      TILE_WIDTH  + MARGIN 

//--------------------------------------------------------------------------
// Output tile buffer dimensions 
//--------------------------------------------------------------------------
#define OUT_BUF_DEPTH     4
#define OUT_BUF_HEIGHT    TILE_HEIGHT / STRIDE
#define OUT_BUF_WIDTH     TILE_WIDTH  / STRIDE

//--------------------------------------------------------------------------
// Top-level Function Declaration
//--------------------------------------------------------------------------
void tiled_conv (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH]
);

#endif
