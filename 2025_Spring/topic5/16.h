
// Command line : python pseudo.py 16 2**255-19

#include <stdint.h>
#include <stdio.h>

#define sspint int16_t
#define spint uint16_t
#define udpint uint32_t
#define dpint uint32_t

#define Wordlength 16
#define Nlimbs 20
#define Radix 13
#define Nbits 255
#define Nbytes 32

#define MERSENNE
#define MULBYINT





static void modmul(const spint *a, const spint *b, spint *c) {
  dpint t = 0;
  dpint tt;
  spint lo;
  dpint hi;
  spint carry;
  spint s;
  spint mask = ((spint)1 << 13u) - (spint)1;
  tt = (dpint)a[1] * (dpint)b[19];
  tt += (dpint)a[2] * (dpint)b[18];
  tt += (dpint)a[3] * (dpint)b[17];
  tt += (dpint)a[4] * (dpint)b[16];
  tt += (dpint)a[5] * (dpint)b[15];
  tt += (dpint)a[6] * (dpint)b[14];
  tt += (dpint)a[7] * (dpint)b[13];
  tt += (dpint)a[8] * (dpint)b[12];
  tt += (dpint)a[9] * (dpint)b[11];
  tt += (dpint)a[10] * (dpint)b[10];
  tt += (dpint)a[11] * (dpint)b[9];
  tt += (dpint)a[12] * (dpint)b[8];
  tt += (dpint)a[13] * (dpint)b[7];
  tt += (dpint)a[14] * (dpint)b[6];
  tt += (dpint)a[15] * (dpint)b[5];
  tt += (dpint)a[16] * (dpint)b[4];
  tt += (dpint)a[17] * (dpint)b[3];
  tt += (dpint)a[18] * (dpint)b[2];
  tt += (dpint)a[19] * (dpint)b[1];
  lo = (spint)tt & mask;
  t += (dpint)lo * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[0];
  spint v0 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[2] * (dpint)b[19];
  tt += (dpint)a[3] * (dpint)b[18];
  tt += (dpint)a[4] * (dpint)b[17];
  tt += (dpint)a[5] * (dpint)b[16];
  tt += (dpint)a[6] * (dpint)b[15];
  tt += (dpint)a[7] * (dpint)b[14];
  tt += (dpint)a[8] * (dpint)b[13];
  tt += (dpint)a[9] * (dpint)b[12];
  tt += (dpint)a[10] * (dpint)b[11];
  tt += (dpint)a[11] * (dpint)b[10];
  tt += (dpint)a[12] * (dpint)b[9];
  tt += (dpint)a[13] * (dpint)b[8];
  tt += (dpint)a[14] * (dpint)b[7];
  tt += (dpint)a[15] * (dpint)b[6];
  tt += (dpint)a[16] * (dpint)b[5];
  tt += (dpint)a[17] * (dpint)b[4];
  tt += (dpint)a[18] * (dpint)b[3];
  tt += (dpint)a[19] * (dpint)b[2];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[1];
  t += (dpint)a[1] * (dpint)b[0];
  spint v1 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[3] * (dpint)b[19];
  tt += (dpint)a[4] * (dpint)b[18];
  tt += (dpint)a[5] * (dpint)b[17];
  tt += (dpint)a[6] * (dpint)b[16];
  tt += (dpint)a[7] * (dpint)b[15];
  tt += (dpint)a[8] * (dpint)b[14];
  tt += (dpint)a[9] * (dpint)b[13];
  tt += (dpint)a[10] * (dpint)b[12];
  tt += (dpint)a[11] * (dpint)b[11];
  tt += (dpint)a[12] * (dpint)b[10];
  tt += (dpint)a[13] * (dpint)b[9];
  tt += (dpint)a[14] * (dpint)b[8];
  tt += (dpint)a[15] * (dpint)b[7];
  tt += (dpint)a[16] * (dpint)b[6];
  tt += (dpint)a[17] * (dpint)b[5];
  tt += (dpint)a[18] * (dpint)b[4];
  tt += (dpint)a[19] * (dpint)b[3];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[2];
  t += (dpint)a[1] * (dpint)b[1];
  t += (dpint)a[2] * (dpint)b[0];
  spint v2 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[4] * (dpint)b[19];
  tt += (dpint)a[5] * (dpint)b[18];
  tt += (dpint)a[6] * (dpint)b[17];
  tt += (dpint)a[7] * (dpint)b[16];
  tt += (dpint)a[8] * (dpint)b[15];
  tt += (dpint)a[9] * (dpint)b[14];
  tt += (dpint)a[10] * (dpint)b[13];
  tt += (dpint)a[11] * (dpint)b[12];
  tt += (dpint)a[12] * (dpint)b[11];
  tt += (dpint)a[13] * (dpint)b[10];
  tt += (dpint)a[14] * (dpint)b[9];
  tt += (dpint)a[15] * (dpint)b[8];
  tt += (dpint)a[16] * (dpint)b[7];
  tt += (dpint)a[17] * (dpint)b[6];
  tt += (dpint)a[18] * (dpint)b[5];
  tt += (dpint)a[19] * (dpint)b[4];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[3];
  t += (dpint)a[1] * (dpint)b[2];
  t += (dpint)a[2] * (dpint)b[1];
  t += (dpint)a[3] * (dpint)b[0];
  spint v3 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[5] * (dpint)b[19];
  tt += (dpint)a[6] * (dpint)b[18];
  tt += (dpint)a[7] * (dpint)b[17];
  tt += (dpint)a[8] * (dpint)b[16];
  tt += (dpint)a[9] * (dpint)b[15];
  tt += (dpint)a[10] * (dpint)b[14];
  tt += (dpint)a[11] * (dpint)b[13];
  tt += (dpint)a[12] * (dpint)b[12];
  tt += (dpint)a[13] * (dpint)b[11];
  tt += (dpint)a[14] * (dpint)b[10];
  tt += (dpint)a[15] * (dpint)b[9];
  tt += (dpint)a[16] * (dpint)b[8];
  tt += (dpint)a[17] * (dpint)b[7];
  tt += (dpint)a[18] * (dpint)b[6];
  tt += (dpint)a[19] * (dpint)b[5];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[4];
  t += (dpint)a[1] * (dpint)b[3];
  t += (dpint)a[2] * (dpint)b[2];
  t += (dpint)a[3] * (dpint)b[1];
  t += (dpint)a[4] * (dpint)b[0];
  spint v4 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[6] * (dpint)b[19];
  tt += (dpint)a[7] * (dpint)b[18];
  tt += (dpint)a[8] * (dpint)b[17];
  tt += (dpint)a[9] * (dpint)b[16];
  tt += (dpint)a[10] * (dpint)b[15];
  tt += (dpint)a[11] * (dpint)b[14];
  tt += (dpint)a[12] * (dpint)b[13];
  tt += (dpint)a[13] * (dpint)b[12];
  tt += (dpint)a[14] * (dpint)b[11];
  tt += (dpint)a[15] * (dpint)b[10];
  tt += (dpint)a[16] * (dpint)b[9];
  tt += (dpint)a[17] * (dpint)b[8];
  tt += (dpint)a[18] * (dpint)b[7];
  tt += (dpint)a[19] * (dpint)b[6];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[5];
  t += (dpint)a[1] * (dpint)b[4];
  t += (dpint)a[2] * (dpint)b[3];
  t += (dpint)a[3] * (dpint)b[2];
  t += (dpint)a[4] * (dpint)b[1];
  t += (dpint)a[5] * (dpint)b[0];
  spint v5 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[7] * (dpint)b[19];
  tt += (dpint)a[8] * (dpint)b[18];
  tt += (dpint)a[9] * (dpint)b[17];
  tt += (dpint)a[10] * (dpint)b[16];
  tt += (dpint)a[11] * (dpint)b[15];
  tt += (dpint)a[12] * (dpint)b[14];
  tt += (dpint)a[13] * (dpint)b[13];
  tt += (dpint)a[14] * (dpint)b[12];
  tt += (dpint)a[15] * (dpint)b[11];
  tt += (dpint)a[16] * (dpint)b[10];
  tt += (dpint)a[17] * (dpint)b[9];
  tt += (dpint)a[18] * (dpint)b[8];
  tt += (dpint)a[19] * (dpint)b[7];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[6];
  t += (dpint)a[1] * (dpint)b[5];
  t += (dpint)a[2] * (dpint)b[4];
  t += (dpint)a[3] * (dpint)b[3];
  t += (dpint)a[4] * (dpint)b[2];
  t += (dpint)a[5] * (dpint)b[1];
  t += (dpint)a[6] * (dpint)b[0];
  spint v6 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[8] * (dpint)b[19];
  tt += (dpint)a[9] * (dpint)b[18];
  tt += (dpint)a[10] * (dpint)b[17];
  tt += (dpint)a[11] * (dpint)b[16];
  tt += (dpint)a[12] * (dpint)b[15];
  tt += (dpint)a[13] * (dpint)b[14];
  tt += (dpint)a[14] * (dpint)b[13];
  tt += (dpint)a[15] * (dpint)b[12];
  tt += (dpint)a[16] * (dpint)b[11];
  tt += (dpint)a[17] * (dpint)b[10];
  tt += (dpint)a[18] * (dpint)b[9];
  tt += (dpint)a[19] * (dpint)b[8];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[7];
  t += (dpint)a[1] * (dpint)b[6];
  t += (dpint)a[2] * (dpint)b[5];
  t += (dpint)a[3] * (dpint)b[4];
  t += (dpint)a[4] * (dpint)b[3];
  t += (dpint)a[5] * (dpint)b[2];
  t += (dpint)a[6] * (dpint)b[1];
  t += (dpint)a[7] * (dpint)b[0];
  spint v7 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[9] * (dpint)b[19];
  tt += (dpint)a[10] * (dpint)b[18];
  tt += (dpint)a[11] * (dpint)b[17];
  tt += (dpint)a[12] * (dpint)b[16];
  tt += (dpint)a[13] * (dpint)b[15];
  tt += (dpint)a[14] * (dpint)b[14];
  tt += (dpint)a[15] * (dpint)b[13];
  tt += (dpint)a[16] * (dpint)b[12];
  tt += (dpint)a[17] * (dpint)b[11];
  tt += (dpint)a[18] * (dpint)b[10];
  tt += (dpint)a[19] * (dpint)b[9];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[8];
  t += (dpint)a[1] * (dpint)b[7];
  t += (dpint)a[2] * (dpint)b[6];
  t += (dpint)a[3] * (dpint)b[5];
  t += (dpint)a[4] * (dpint)b[4];
  t += (dpint)a[5] * (dpint)b[3];
  t += (dpint)a[6] * (dpint)b[2];
  t += (dpint)a[7] * (dpint)b[1];
  t += (dpint)a[8] * (dpint)b[0];
  spint v8 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[10] * (dpint)b[19];
  tt += (dpint)a[11] * (dpint)b[18];
  tt += (dpint)a[12] * (dpint)b[17];
  tt += (dpint)a[13] * (dpint)b[16];
  tt += (dpint)a[14] * (dpint)b[15];
  tt += (dpint)a[15] * (dpint)b[14];
  tt += (dpint)a[16] * (dpint)b[13];
  tt += (dpint)a[17] * (dpint)b[12];
  tt += (dpint)a[18] * (dpint)b[11];
  tt += (dpint)a[19] * (dpint)b[10];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[9];
  t += (dpint)a[1] * (dpint)b[8];
  t += (dpint)a[2] * (dpint)b[7];
  t += (dpint)a[3] * (dpint)b[6];
  t += (dpint)a[4] * (dpint)b[5];
  t += (dpint)a[5] * (dpint)b[4];
  t += (dpint)a[6] * (dpint)b[3];
  t += (dpint)a[7] * (dpint)b[2];
  t += (dpint)a[8] * (dpint)b[1];
  t += (dpint)a[9] * (dpint)b[0];
  spint v9 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[11] * (dpint)b[19];
  tt += (dpint)a[12] * (dpint)b[18];
  tt += (dpint)a[13] * (dpint)b[17];
  tt += (dpint)a[14] * (dpint)b[16];
  tt += (dpint)a[15] * (dpint)b[15];
  tt += (dpint)a[16] * (dpint)b[14];
  tt += (dpint)a[17] * (dpint)b[13];
  tt += (dpint)a[18] * (dpint)b[12];
  tt += (dpint)a[19] * (dpint)b[11];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[10];
  t += (dpint)a[1] * (dpint)b[9];
  t += (dpint)a[2] * (dpint)b[8];
  t += (dpint)a[3] * (dpint)b[7];
  t += (dpint)a[4] * (dpint)b[6];
  t += (dpint)a[5] * (dpint)b[5];
  t += (dpint)a[6] * (dpint)b[4];
  t += (dpint)a[7] * (dpint)b[3];
  t += (dpint)a[8] * (dpint)b[2];
  t += (dpint)a[9] * (dpint)b[1];
  t += (dpint)a[10] * (dpint)b[0];
  spint v10 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[12] * (dpint)b[19];
  tt += (dpint)a[13] * (dpint)b[18];
  tt += (dpint)a[14] * (dpint)b[17];
  tt += (dpint)a[15] * (dpint)b[16];
  tt += (dpint)a[16] * (dpint)b[15];
  tt += (dpint)a[17] * (dpint)b[14];
  tt += (dpint)a[18] * (dpint)b[13];
  tt += (dpint)a[19] * (dpint)b[12];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[11];
  t += (dpint)a[1] * (dpint)b[10];
  t += (dpint)a[2] * (dpint)b[9];
  t += (dpint)a[3] * (dpint)b[8];
  t += (dpint)a[4] * (dpint)b[7];
  t += (dpint)a[5] * (dpint)b[6];
  t += (dpint)a[6] * (dpint)b[5];
  t += (dpint)a[7] * (dpint)b[4];
  t += (dpint)a[8] * (dpint)b[3];
  t += (dpint)a[9] * (dpint)b[2];
  t += (dpint)a[10] * (dpint)b[1];
  t += (dpint)a[11] * (dpint)b[0];
  spint v11 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[13] * (dpint)b[19];
  tt += (dpint)a[14] * (dpint)b[18];
  tt += (dpint)a[15] * (dpint)b[17];
  tt += (dpint)a[16] * (dpint)b[16];
  tt += (dpint)a[17] * (dpint)b[15];
  tt += (dpint)a[18] * (dpint)b[14];
  tt += (dpint)a[19] * (dpint)b[13];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[12];
  t += (dpint)a[1] * (dpint)b[11];
  t += (dpint)a[2] * (dpint)b[10];
  t += (dpint)a[3] * (dpint)b[9];
  t += (dpint)a[4] * (dpint)b[8];
  t += (dpint)a[5] * (dpint)b[7];
  t += (dpint)a[6] * (dpint)b[6];
  t += (dpint)a[7] * (dpint)b[5];
  t += (dpint)a[8] * (dpint)b[4];
  t += (dpint)a[9] * (dpint)b[3];
  t += (dpint)a[10] * (dpint)b[2];
  t += (dpint)a[11] * (dpint)b[1];
  t += (dpint)a[12] * (dpint)b[0];
  spint v12 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[14] * (dpint)b[19];
  tt += (dpint)a[15] * (dpint)b[18];
  tt += (dpint)a[16] * (dpint)b[17];
  tt += (dpint)a[17] * (dpint)b[16];
  tt += (dpint)a[18] * (dpint)b[15];
  tt += (dpint)a[19] * (dpint)b[14];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[13];
  t += (dpint)a[1] * (dpint)b[12];
  t += (dpint)a[2] * (dpint)b[11];
  t += (dpint)a[3] * (dpint)b[10];
  t += (dpint)a[4] * (dpint)b[9];
  t += (dpint)a[5] * (dpint)b[8];
  t += (dpint)a[6] * (dpint)b[7];
  t += (dpint)a[7] * (dpint)b[6];
  t += (dpint)a[8] * (dpint)b[5];
  t += (dpint)a[9] * (dpint)b[4];
  t += (dpint)a[10] * (dpint)b[3];
  t += (dpint)a[11] * (dpint)b[2];
  t += (dpint)a[12] * (dpint)b[1];
  t += (dpint)a[13] * (dpint)b[0];
  spint v13 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[15] * (dpint)b[19];
  tt += (dpint)a[16] * (dpint)b[18];
  tt += (dpint)a[17] * (dpint)b[17];
  tt += (dpint)a[18] * (dpint)b[16];
  tt += (dpint)a[19] * (dpint)b[15];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[14];
  t += (dpint)a[1] * (dpint)b[13];
  t += (dpint)a[2] * (dpint)b[12];
  t += (dpint)a[3] * (dpint)b[11];
  t += (dpint)a[4] * (dpint)b[10];
  t += (dpint)a[5] * (dpint)b[9];
  t += (dpint)a[6] * (dpint)b[8];
  t += (dpint)a[7] * (dpint)b[7];
  t += (dpint)a[8] * (dpint)b[6];
  t += (dpint)a[9] * (dpint)b[5];
  t += (dpint)a[10] * (dpint)b[4];
  t += (dpint)a[11] * (dpint)b[3];
  t += (dpint)a[12] * (dpint)b[2];
  t += (dpint)a[13] * (dpint)b[1];
  t += (dpint)a[14] * (dpint)b[0];
  spint v14 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[16] * (dpint)b[19];
  tt += (dpint)a[17] * (dpint)b[18];
  tt += (dpint)a[18] * (dpint)b[17];
  tt += (dpint)a[19] * (dpint)b[16];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[15];
  t += (dpint)a[1] * (dpint)b[14];
  t += (dpint)a[2] * (dpint)b[13];
  t += (dpint)a[3] * (dpint)b[12];
  t += (dpint)a[4] * (dpint)b[11];
  t += (dpint)a[5] * (dpint)b[10];
  t += (dpint)a[6] * (dpint)b[9];
  t += (dpint)a[7] * (dpint)b[8];
  t += (dpint)a[8] * (dpint)b[7];
  t += (dpint)a[9] * (dpint)b[6];
  t += (dpint)a[10] * (dpint)b[5];
  t += (dpint)a[11] * (dpint)b[4];
  t += (dpint)a[12] * (dpint)b[3];
  t += (dpint)a[13] * (dpint)b[2];
  t += (dpint)a[14] * (dpint)b[1];
  t += (dpint)a[15] * (dpint)b[0];
  spint v15 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[17] * (dpint)b[19];
  tt += (dpint)a[18] * (dpint)b[18];
  tt += (dpint)a[19] * (dpint)b[17];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[16];
  t += (dpint)a[1] * (dpint)b[15];
  t += (dpint)a[2] * (dpint)b[14];
  t += (dpint)a[3] * (dpint)b[13];
  t += (dpint)a[4] * (dpint)b[12];
  t += (dpint)a[5] * (dpint)b[11];
  t += (dpint)a[6] * (dpint)b[10];
  t += (dpint)a[7] * (dpint)b[9];
  t += (dpint)a[8] * (dpint)b[8];
  t += (dpint)a[9] * (dpint)b[7];
  t += (dpint)a[10] * (dpint)b[6];
  t += (dpint)a[11] * (dpint)b[5];
  t += (dpint)a[12] * (dpint)b[4];
  t += (dpint)a[13] * (dpint)b[3];
  t += (dpint)a[14] * (dpint)b[2];
  t += (dpint)a[15] * (dpint)b[1];
  t += (dpint)a[16] * (dpint)b[0];
  spint v16 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[18] * (dpint)b[19];
  tt += (dpint)a[19] * (dpint)b[18];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[17];
  t += (dpint)a[1] * (dpint)b[16];
  t += (dpint)a[2] * (dpint)b[15];
  t += (dpint)a[3] * (dpint)b[14];
  t += (dpint)a[4] * (dpint)b[13];
  t += (dpint)a[5] * (dpint)b[12];
  t += (dpint)a[6] * (dpint)b[11];
  t += (dpint)a[7] * (dpint)b[10];
  t += (dpint)a[8] * (dpint)b[9];
  t += (dpint)a[9] * (dpint)b[8];
  t += (dpint)a[10] * (dpint)b[7];
  t += (dpint)a[11] * (dpint)b[6];
  t += (dpint)a[12] * (dpint)b[5];
  t += (dpint)a[13] * (dpint)b[4];
  t += (dpint)a[14] * (dpint)b[3];
  t += (dpint)a[15] * (dpint)b[2];
  t += (dpint)a[16] * (dpint)b[1];
  t += (dpint)a[17] * (dpint)b[0];
  spint v17 = (spint)t & mask;
  t = t >> 13u;
  tt = (dpint)a[19] * (dpint)b[19];
  lo = (spint)tt & mask;
  t += (hi + (dpint)lo) * (dpint)0x260;
  hi = tt >> 13u;
  t += (dpint)a[0] * (dpint)b[18];
  t += (dpint)a[1] * (dpint)b[17];
  t += (dpint)a[2] * (dpint)b[16];
  t += (dpint)a[3] * (dpint)b[15];
  t += (dpint)a[4] * (dpint)b[14];
  t += (dpint)a[5] * (dpint)b[13];
  t += (dpint)a[6] * (dpint)b[12];
  t += (dpint)a[7] * (dpint)b[11];
  t += (dpint)a[8] * (dpint)b[10];
  t += (dpint)a[9] * (dpint)b[9];
  t += (dpint)a[10] * (dpint)b[8];
  t += (dpint)a[11] * (dpint)b[7];
  t += (dpint)a[12] * (dpint)b[6];
  t += (dpint)a[13] * (dpint)b[5];
  t += (dpint)a[14] * (dpint)b[4];
  t += (dpint)a[15] * (dpint)b[3];
  t += (dpint)a[16] * (dpint)b[2];
  t += (dpint)a[17] * (dpint)b[1];
  t += (dpint)a[18] * (dpint)b[0];
  spint v18 = (spint)t & mask;
  t = t >> 13u;
  t += (dpint)a[0] * (dpint)b[19];
  t += (dpint)a[1] * (dpint)b[18];
  t += (dpint)a[2] * (dpint)b[17];
  t += (dpint)a[3] * (dpint)b[16];
  t += (dpint)a[4] * (dpint)b[15];
  t += (dpint)a[5] * (dpint)b[14];
  t += (dpint)a[6] * (dpint)b[13];
  t += (dpint)a[7] * (dpint)b[12];
  t += (dpint)a[8] * (dpint)b[11];
  t += (dpint)a[9] * (dpint)b[10];
  t += (dpint)a[10] * (dpint)b[9];
  t += (dpint)a[11] * (dpint)b[8];
  t += (dpint)a[12] * (dpint)b[7];
  t += (dpint)a[13] * (dpint)b[6];
  t += (dpint)a[14] * (dpint)b[5];
  t += (dpint)a[15] * (dpint)b[4];
  t += (dpint)a[16] * (dpint)b[3];
  t += (dpint)a[17] * (dpint)b[2];
  t += (dpint)a[18] * (dpint)b[1];
  t += (dpint)a[19] * (dpint)b[0];
  t += (dpint)hi * (dpint)0x260;
  spint v19 = (spint)t & mask;
  t = t >> 13u;
  // second reduction pass

  udpint ut = (udpint)t;
  ut = (ut << 5) + (spint)(v19 >> 8u);
  v19 &= 0xff;
  ut *= 0x13;
  s = v0 + ((spint)ut & mask);
  c[0] = (spint)(s & mask);
  ut = (udpint)(s >> 13) + (ut >> 13);
  s = v1 + ((spint)ut & mask);
  c[1] = (spint)(s & mask);
  carry = (s >> 13) + (spint)(ut >> 13);
  c[2] = v2 + carry;
  c[3] = v3;
  c[4] = v4;
  c[5] = v5;
  c[6] = v6;
  c[7] = v7;
  c[8] = v8;
  c[9] = v9;
  c[10] = v10;
  c[11] = v11;
  c[12] = v12;
  c[13] = v13;
  c[14] = v14;
  c[15] = v15;
  c[16] = v16;
  c[17] = v17;
  c[18] = v18;
  c[19] = v19;
}
