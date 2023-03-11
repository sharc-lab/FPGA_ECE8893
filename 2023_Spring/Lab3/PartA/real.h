///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    real.h
// Description: Header file for real matrix multiplication
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////

#ifndef __REAL_H__
#define __REAL_H__

#include <stdio.h>
#include <stdlib.h>

#include <ap_int.h>

typedef ap_int<16> real_t;

#define M 100
#define N 150
#define K 200

void real_matmul( 
    real_t MatA_DRAM[M][N], 
    real_t MatB_DRAM[N][K], 
    real_t MatC_DRAM[M][K]
);

#endif
