///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    model_conv.cpp
// Description: Create C model convolution for functional correctness.
//
// TODO: Implement the 7x7 convolution any way you like!
///////////////////////////////////////////////////////////////////////////////

#include "conv.h"

void model_conv (
    fm_t input_feature_map[3][736][1280],
    wt_t layer_weights[64][3][7][7],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][368][640]
)
{
//--------------------------------------------------------------------------
// Your code for TASK A goes here 
//
// Implement the 7x7 convolution layer in typical C/C++ manner.
// You are free to use any C/C++ programming constructs which may
// or may not be HLS-friendly.
//
// The sole purpose of this code is to get the functionality right!
//
// Hints: 
// - Handle stride and border pixels appropriately.
// - Do not forget to add bias and apply ReLU!
//--------------------------------------------------------------------------
}
