#ifndef DCL_H
#define DCL_H

#include <ap_fixed.h>
#include <cstdint>

// Total number of elements (must be divisible by BLOCK)
#define N (1 << 16)      // 65536
#define BLOCK 256        // block size for stats (rate mismatch)

// Fixed-point types
typedef ap_fixed<20, 6, AP_RND, AP_SAT> data_t;   // main signal
typedef ap_fixed<36, 12, AP_RND, AP_SAT> acc_t;   // accumulator for reductions
typedef ap_fixed<28, 10, AP_RND, AP_SAT> stat_t;  // per-block statistic
typedef ap_fixed<28, 10, AP_RND, AP_SAT> coef_t;  // coefficients

// Top-level kernel prototype
void top_kernel(const data_t in[N],
                data_t out[N]);

#endif // DCL_H
