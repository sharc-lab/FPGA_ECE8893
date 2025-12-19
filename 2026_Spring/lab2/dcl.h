#ifndef DCL_H
#define DCL_H

#include <cstdint>
#include <ap_fixed.h>

// Grid size and timesteps (fixed for fair ranking)
#define NX 256
#define NY 256
#define TSTEPS 30

// Fixed-point types
// data_t: stored grid values
// acc_t: wider accumulator for weighted sums
typedef ap_fixed<24, 8, AP_RND, AP_SAT> data_t;
typedef ap_fixed<40, 12, AP_RND, AP_SAT> acc_t;

// Top-level kernel prototype
void top_kernel(const data_t A_in[NX][NY],
                data_t A_out[NX][NY]);

#endif // DCL_H
