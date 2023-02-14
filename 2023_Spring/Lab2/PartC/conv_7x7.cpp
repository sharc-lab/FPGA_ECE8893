///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Implement an optimized 7x7 convolution for a single tile block
//              
// TODO: Use your unoptimized code from Part B and apply your favorite pragmas
//       to achieve the target latency (or lower)!
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
// Part C: Optimize your Part B code to achieve an overall latency of 
//         less than 750ms without exceeding 100% resource utiliization.
//
// TODO: Your code for Part C goes here. 
//---------------------------------------------------------------------------

}
