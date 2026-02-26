#include "dcl.h"
#include "hls_stream.h"
#include <ap_int.h>

static constexpr int AXI_W     = 512;
static constexpr int AXI_ELEMS = AXI_W / 32;
static constexpr int P         = 8;
using axi_t = ap_uint<AXI_W>;

static_assert(NY % P == 0, "NY must be divisible by P");
static constexpr int NG = NY / P;

// Narrow types: 18×25 fits one DSP48E1
typedef ap_fixed<18, 2, AP_TRN, AP_WRAP> narrow_coeff_t;
typedef ap_fixed<18, 2, AP_TRN, AP_WRAP> narrow_sum_t;
typedef ap_fixed<19, 3, AP_TRN, AP_WRAP> wide_sum_t;

// ---- helpers ----
static data_t lane32_to_data(ap_uint<32> x) {
#pragma HLS inline
    data_t v;
    v.range(23, 0) = x.range(23, 0);
    return v;
}

static ap_uint<32> data_to_lane32(data_t v) {
#pragma HLS inline
    ap_uint<32> out = 0;
    out.range(23, 0) = v.range(23, 0);
    return out; 
}
struct vec_t { narrow_sum_t d[P]; };

// ---------------------------------------------------------------------------
static void load_wide(const data_t A_in[NX][NY], hls::stream<axi_t>& wstrm) {
#pragma HLS inline off
    const axi_t* in_wide = reinterpret_cast<const axi_t*>(&A_in[0][0]);
    const int WORDS = (NX * NY) / AXI_ELEMS;
READ_WIDE:
    for (int w = 0; w < WORDS; w++) {
#pragma HLS pipeline II=1
        wstrm.write(in_wide[w]);
    }
}

// ---------------------------------------------------------------------------
static void wide_to_vec(hls::stream<axi_t>& wstrm, hls::stream<vec_t>& vs) {
#pragma HLS inline off
    const int WORDS = (NX * NY) / AXI_ELEMS;
W2V:
    for (int w = 0; w < WORDS; w++) {
        axi_t memw = wstrm.read();
        for (int g = 0; g < AXI_ELEMS / P; g++) {
#pragma HLS pipeline II=1
            vec_t v;
            for (int k = 0; k < P; k++) {
#pragma HLS unroll
                int lane = g * P + k;
                ap_uint<32> bits = memw.range((lane + 1) * 32 - 1, lane * 32);
                v.d[k] = lane32_to_data(bits);
            }
            vs.write(v);
        }
    }
}

// ---------------------------------------------------------------------------
template<int STAGE>
static void stencil_stage_par(hls::stream<vec_t>& in, hls::stream<vec_t>& out) {
#pragma HLS inline off

    const narrow_coeff_t narrow_wa = (narrow_coeff_t)0.10;
    const narrow_coeff_t narrow_wd = (narrow_coeff_t)0.025;

static vec_t line0[NG];
static vec_t line1[NG];
#pragma HLS bind_storage variable=line0 type=ram_s2p impl=bram
#pragma HLS bind_storage variable=line1 type=ram_s2p impl=bram
#pragma HLS aggregate variable=line0 compact=bit
#pragma HLS aggregate variable=line1 compact=bit
#pragma HLS reset variable=line0 off
#pragma HLS reset variable=line1 off

    narrow_sum_t prev_top[P], prev_mid[P], prev_bot[P];
#pragma HLS array_partition variable=prev_top complete
#pragma HLS array_partition variable=prev_mid complete
#pragma HLS array_partition variable=prev_bot complete

    narrow_sum_t pprev_top_last, pprev_mid_last, pprev_bot_last;

#pragma HLS reset variable=prev_top off
#pragma HLS reset variable=prev_mid off  
#pragma HLS reset variable=prev_bot off
#pragma HLS reset variable=pprev_top_last off
#pragma HLS reset variable=pprev_mid_last off
#pragma HLS reset variable=pprev_bot_last off

ROW:
    for (int i = 0; i <= NX; i++) {

        for (int k = 0; k < P; k++) {
#pragma HLS unroll
            prev_top[k] = 0; prev_mid[k] = 0; prev_bot[k] = 0;
        }
        pprev_top_last = 0; pprev_mid_last = 0; pprev_bot_last = 0;

COL_GROUP:
        for (int jg = 0; jg <= NG; jg++) {
#pragma HLS pipeline II=1
//#pragma HLS dependence variable=line0 inter false
//#pragma HLS dependence variable=line1 inter false

            const bool has_input = (i < NX) && (jg < NG);
            const bool has_lbuf  = (jg < NG);
            const int j_base = jg * P;

            vec_t cur_topv, cur_midv, cur_botv;
            #pragma HLS aggregate variable=cur_topv compact=bit
            #pragma HLS aggregate variable=cur_midv compact=bit
            #pragma HLS aggregate variable=cur_botv compact=bit

            // defaults
            for (int k = 0; k < P; k++) {
            #pragma HLS unroll
                cur_topv.d[k] = 0;
                cur_midv.d[k] = 0;
                cur_botv.d[k] = 0;
            }

            if (has_lbuf) {
                vec_t top_rd = line0[jg];
                vec_t mid_rd = line1[jg];
                cur_topv = top_rd;
                cur_midv = mid_rd;
                if (has_input) {
                    cur_botv = in.read();
                    line0[jg] = mid_rd;
                    line1[jg] = cur_botv;
                }
            }

            if (i >= 1 && jg >= 1) {
                const int oi = i - 1;
                const int oj_base = (jg - 1) * P;
                bool row_boundary = (oi == 0 || oi == NX - 1);
                bool col_boundary_lo = (jg == 1); // oj_base == 0
                bool col_boundary_hi = (jg == NG); // oj_base + P - 1 == NY - 1
                vec_t vout;
                for (int k = 0; k < P; k++) {
#pragma HLS unroll
                    const int oj = oj_base + k;
                    narrow_sum_t center = prev_mid[k];
                    bool is_boundary = row_boundary || (col_boundary_lo && k == 0) 
                                || (col_boundary_hi && k == P-1);
                    narrow_sum_t w01 = prev_top[k];
                    narrow_sum_t w21 = prev_bot[k];
                    narrow_sum_t w10 = (k == 0) ? pprev_mid_last : prev_mid[k - 1];
                    narrow_sum_t w00 = (k == 0) ? pprev_top_last : prev_top[k - 1];
                    narrow_sum_t w20 = (k == 0) ? pprev_bot_last : prev_bot[k - 1];
                    narrow_sum_t w12 = (k == P - 1) ? cur_midv.d[0] : prev_mid[k + 1];
                    narrow_sum_t w02 = (k == P - 1) ? cur_topv.d[0] : prev_top[k + 1];
                    narrow_sum_t w22 = (k == P - 1) ? cur_botv.d[0] : prev_bot[k + 1];
                    wide_sum_t t0 = (wide_sum_t)w00 + (wide_sum_t)w02;
                    wide_sum_t t1 = (wide_sum_t)w20 + (wide_sum_t)w22;
                    wide_sum_t t2 = (wide_sum_t)w01 + (wide_sum_t)w21;
                    wide_sum_t t3 = (wide_sum_t)w10 + (wide_sum_t)w12;
                    wide_sum_t ns_diag = t0 + t1;
                    wide_sum_t ns_axis = t2 + t3;
                    wide_sum_t outv = ((wide_sum_t)center >> 1)
                                    + (wide_sum_t)narrow_wa * ns_axis
                                    + (wide_sum_t)narrow_wd * ns_diag;
                    vout.d[k] = is_boundary ? center : (narrow_sum_t)outv;
                }
                out.write(vout);
            }

            pprev_top_last = prev_top[P - 1];
            pprev_mid_last = prev_mid[P - 1];
            pprev_bot_last = prev_bot[P - 1];
            for (int k = 0; k < P; k++) {
            #pragma HLS unroll
                prev_top[k] = cur_topv.d[k];
                prev_mid[k] = cur_midv.d[k];
                prev_bot[k] = cur_botv.d[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
static void vec_to_wide(hls::stream<vec_t>& vs, hls::stream<axi_t>& wstrm) {
#pragma HLS inline off
    const int WORDS = (NX * NY) / AXI_ELEMS;
V2W:
    for (int w = 0; w < WORDS; w++) {
        axi_t memw = 0;
        for (int g = 0; g < AXI_ELEMS / P; g++) {
#pragma HLS pipeline II=1
            vec_t v = vs.read();
            for (int k = 0; k < P; k++) {
#pragma HLS unroll
                int lane = g * P + k;
                ap_uint<32> bits = data_to_lane32(v.d[k]);
                memw.range((lane + 1) * 32 - 1, lane * 32) = bits;
            }
        }
        wstrm.write(memw);
    }
}

// ---------------------------------------------------------------------------
static void store_wide(hls::stream<axi_t>& wstrm, data_t A_out[NX][NY]) {
#pragma HLS inline off
    axi_t* out_wide = reinterpret_cast<axi_t*>(&A_out[0][0]);
    const int WORDS = (NX * NY) / AXI_ELEMS;
WRITE_WIDE:
    for (int w = 0; w < WORDS; w++) {
#pragma HLS pipeline II=1
        out_wide[w] = wstrm.read();
    }
}

// ---------------------------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
#pragma HLS interface m_axi port=A_in  offset=slave bundle=A_in  depth=NX*NY \
  max_read_burst_length=64 num_read_outstanding=16 max_widen_bitwidth=512
#pragma HLS interface m_axi port=A_out offset=slave bundle=A_out depth=NX*NY \
  max_write_burst_length=64 num_write_outstanding=16 max_widen_bitwidth=512
#pragma HLS interface s_axilite port=return

    hls::stream<axi_t> in_wide_s("in_wide_s");
    hls::stream<axi_t> out_wide_s("out_wide_s");
#pragma HLS STREAM variable=in_wide_s  depth=64
#pragma HLS STREAM variable=out_wide_s depth=64

    hls::stream<vec_t> s0("s0");   hls::stream<vec_t> s1("s1");
    hls::stream<vec_t> s2("s2");   hls::stream<vec_t> s3("s3");
    hls::stream<vec_t> s4("s4");   hls::stream<vec_t> s5("s5");
    hls::stream<vec_t> s6("s6");   hls::stream<vec_t> s7("s7");
    hls::stream<vec_t> s8("s8");   hls::stream<vec_t> s9("s9");
    hls::stream<vec_t> s10("s10"); hls::stream<vec_t> s11("s11");
    hls::stream<vec_t> s12("s12"); hls::stream<vec_t> s13("s13");
    hls::stream<vec_t> s14("s14"); hls::stream<vec_t> s15("s15");
    hls::stream<vec_t> s16("s16"); hls::stream<vec_t> s17("s17");
    hls::stream<vec_t> s18("s18"); hls::stream<vec_t> s19("s19");
    hls::stream<vec_t> s20("s20"); hls::stream<vec_t> s21("s21");
    hls::stream<vec_t> s22("s22"); hls::stream<vec_t> s23("s23");
    hls::stream<vec_t> s24("s24"); hls::stream<vec_t> s25("s25");
    hls::stream<vec_t> s26("s26"); hls::stream<vec_t> s27("s27");
    hls::stream<vec_t> s28("s28"); hls::stream<vec_t> s29("s29");
    hls::stream<vec_t> s30("s30");

#pragma HLS STREAM variable=s0  depth=2
#pragma HLS STREAM variable=s1  depth=2
#pragma HLS STREAM variable=s2  depth=2
#pragma HLS STREAM variable=s3  depth=2
#pragma HLS STREAM variable=s4  depth=2
#pragma HLS STREAM variable=s5  depth=2
#pragma HLS STREAM variable=s6  depth=2
#pragma HLS STREAM variable=s7  depth=2
#pragma HLS STREAM variable=s8  depth=2
#pragma HLS STREAM variable=s9  depth=2
#pragma HLS STREAM variable=s10 depth=2
#pragma HLS STREAM variable=s11 depth=2
#pragma HLS STREAM variable=s12 depth=2
#pragma HLS STREAM variable=s13 depth=2
#pragma HLS STREAM variable=s14 depth=2
#pragma HLS STREAM variable=s15 depth=2
#pragma HLS STREAM variable=s16 depth=2
#pragma HLS STREAM variable=s17 depth=2
#pragma HLS STREAM variable=s18 depth=2
#pragma HLS STREAM variable=s19 depth=2
#pragma HLS STREAM variable=s20 depth=2
#pragma HLS STREAM variable=s21 depth=2
#pragma HLS STREAM variable=s22 depth=2
#pragma HLS STREAM variable=s23 depth=2
#pragma HLS STREAM variable=s24 depth=2
#pragma HLS STREAM variable=s25 depth=2
#pragma HLS STREAM variable=s26 depth=2
#pragma HLS STREAM variable=s27 depth=2
#pragma HLS STREAM variable=s28 depth=2
#pragma HLS STREAM variable=s29 depth=2
#pragma HLS STREAM variable=s30 depth=2

#pragma HLS reset variable=s0 off
#pragma HLS reset variable=s1 off
#pragma HLS reset variable=s2 off
#pragma HLS reset variable=s3 off
#pragma HLS reset variable=s4 off
#pragma HLS reset variable=s5 off
#pragma HLS reset variable=s6 off
#pragma HLS reset variable=s7 off
#pragma HLS reset variable=s8 off
#pragma HLS reset variable=s9 off
#pragma HLS reset variable=s10 off
#pragma HLS reset variable=s11 off
#pragma HLS reset variable=s12 off
#pragma HLS reset variable=s13 off
#pragma HLS reset variable=s14 off
#pragma HLS reset variable=s15 off
#pragma HLS reset variable=s16 off
#pragma HLS reset variable=s17 off
#pragma HLS reset variable=s18 off
#pragma HLS reset variable=s19 off
#pragma HLS reset variable=s20 off
#pragma HLS reset variable=s21 off
#pragma HLS reset variable=s22 off
#pragma HLS reset variable=s23 off
#pragma HLS reset variable=s24 off
#pragma HLS reset variable=s25 off
#pragma HLS reset variable=s26 off
#pragma HLS reset variable=s27 off
#pragma HLS reset variable=s28 off
#pragma HLS reset variable=s29 off
#pragma HLS reset variable=s30 off

#pragma HLS dataflow

    load_wide(A_in, in_wide_s);
    wide_to_vec(in_wide_s, s0);

    stencil_stage_par<0>(s0,  s1);   stencil_stage_par<1>(s1,  s2);
    stencil_stage_par<2>(s2,  s3);   stencil_stage_par<3>(s3,  s4);
    stencil_stage_par<4>(s4,  s5);   stencil_stage_par<5>(s5,  s6);
    stencil_stage_par<6>(s6,  s7);   stencil_stage_par<7>(s7,  s8);
    stencil_stage_par<8>(s8,  s9);   stencil_stage_par<9>(s9,  s10);
    stencil_stage_par<10>(s10, s11); stencil_stage_par<11>(s11, s12);
    stencil_stage_par<12>(s12, s13); stencil_stage_par<13>(s13, s14);
    stencil_stage_par<14>(s14, s15); stencil_stage_par<15>(s15, s16);
    stencil_stage_par<16>(s16, s17); stencil_stage_par<17>(s17, s18);
    stencil_stage_par<18>(s18, s19); stencil_stage_par<19>(s19, s20);
    stencil_stage_par<20>(s20, s21); stencil_stage_par<21>(s21, s22);
    stencil_stage_par<22>(s22, s23); stencil_stage_par<23>(s23, s24);
    stencil_stage_par<24>(s24, s25); stencil_stage_par<25>(s25, s26);
    stencil_stage_par<26>(s26, s27); stencil_stage_par<27>(s27, s28);
    stencil_stage_par<28>(s28, s29); stencil_stage_par<29>(s29, s30);

    vec_to_wide(s30, out_wide_s);
    store_wide(out_wide_s, A_out);
}