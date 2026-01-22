#ifndef DCL_H
#define DCL_H

#include <cstdint>
#include <ap_fixed.h>

//
// Problem size (you may adjust)
//
#define N_ROWS 256
#define N_COLS 64

// Fixed-point types
// data_t: stored grid values
// acc_t: wider accumulator for weighted sums
typedef ap_fixed<24, 10, AP_RND, AP_SAT> data_t;

//
// Top-level kernel prototype
//
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]);

#endif // DCL_H
