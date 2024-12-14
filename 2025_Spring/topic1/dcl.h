
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <random>
#include <iostream>
#include <fstream>

#include "ap_fixed.h"

#define DIM 200 // matrix multiplication size
typedef ap_uint<8> data_8;
typedef ap_uint<16> data_16;


// standard 32-bit floating point bitwidth
#define std_FP32_TOTAL_BIT      32
#define std_FP32_SIGN_BIT       1
#define std_FP32_EXPN_BIT       8
#define std_FP32_MANT_BIT       23
#define std_FP32_MANT_MUL_BIT   48
#define std_FP32_BIAS           127

#define myFP_MAX_TOTAL_BIT      8
#define myFP_MAX_SIGN_BIT       1
#define myFP_MAX_EXPN_BIT       6 // in testing, exp can go up to 6, although standard is 5-bit
#define myFP_MAX_MANT_BIT       13 // in testing, mant can go up to 13, although standard is 10-bit
#define myFP_MAX_MANT_MUL_BIT   28

typedef ap_uint<myFP_MAX_TOTAL_BIT> myFP;
typedef ap_uint<myFP_MAX_SIGN_BIT> myFP_sign;
typedef ap_uint<myFP_MAX_EXPN_BIT + 2> myFP_expn; // account for exp overflow
typedef ap_uint<myFP_MAX_MANT_BIT + 1> myFP_mant; // account for the implicit 1 before mantissa


float randomFloatRange(float min1, float max1, float min2, float max2);
void float2myFP(float *f, myFP *h, int EB, int MB, bool *overflow);
float myFP2float(const myFP h, int EB, int MB);

void MatMul_E5M2( data_8 a[DIM][DIM], data_8 b[DIM][DIM], data_8 c[DIM][DIM]);
void MatMul_E4M3( data_8 a[DIM][DIM], data_8 b[DIM][DIM], data_8 c[DIM][DIM]);
void MatMul_E5M10( data_16 a[DIM][DIM], data_16 b[DIM][DIM], data_16 c[DIM][DIM]);
void MatMul_Int_8( data_8 a[DIM][DIM], data_8 b[DIM][DIM], data_8 c[DIM][DIM]);
void MatMul_AP_fix_16_5( data_16 a[DIM][DIM], data_16 b[DIM][DIM], data_16 c[DIM][DIM]);
void MatMul_AP_fix_8_4( data_8 a[DIM][DIM], data_8 b[DIM][DIM], data_8 c[DIM][DIM]);

