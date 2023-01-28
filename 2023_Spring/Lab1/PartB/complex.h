///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    complex.h
// Description: Header file for complex matrix multiplication
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////
#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include <ap_int.h>

typedef ap_int<16> real_t;

// Issue: https://piazza.com/class/lczabeu4s8s1io/post/22
// Uses the C++ standard alignas type specifier to specify custom alignment of variables and user defined types.
// Reference: https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Impact-of-Struct-Size-on-Pipelining
struct alignas(8) complex_t {
    real_t real;
    real_t imag;
};

#define M 100
#define N 150
#define K 200

void complex_matmul ( 
    complex_t MatA_DRAM[M][N], 
    complex_t MatB_DRAM[N][K], 
    complex_t MatC_DRAM[M][K]
);

using namespace std;

#endif
