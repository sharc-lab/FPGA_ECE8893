///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Implement a functionally-correct synthesizable 7x7 convolution 
//              for a single tile block without any optimizations
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void conv_7x7 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH]
)
{
//---------------------------------------------------------------------------
// Part B: Implement a trivial functionally-correct single-tile convolution.
//
//         This should have an overall latency in the order of 22-23 seconds.
//
//         If it's worse than trivial, it may be worth fixing this first.
//         Otherwise, achieving the target latency with a worse-than-trivial
//         baseline may be difficult!
//
// TODO: Your code for Part B goes here. 
//---------------------------------------------------------------------------

}
