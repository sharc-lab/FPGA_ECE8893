///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv.h
// Description: Header file for C model simulation of the first 7x7 conv layer 
//              of ResNet-50 DNN
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////

#ifndef CONV_H_
#define CONV_H_

#include <iostream>

typedef float fm_t;
typedef float wt_t;

#define IN_FM_DEPTH        3
#define IN_FM_HEIGHT     736
#define IN_FM_WIDTH     1280

#define OUT_FM_DEPTH      64
#define OUT_FM_HEIGHT    368
#define OUT_FM_WIDTH     640

#define STRIDE             2
#define PADDING            3 

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------
void model_conv (
    fm_t input_feature_map[3][736][1280],
    wt_t layer_weights[64][3][7][7],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][368][640]
);

#endif
