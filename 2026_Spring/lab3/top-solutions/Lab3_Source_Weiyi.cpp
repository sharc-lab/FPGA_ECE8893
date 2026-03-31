#include "dcl.h"
#include <ap_int.h>
#include <hls_stream.h>

static constexpr int VEC = 32;
static constexpr int NUM_BLOCKS = N / BLOCK;
static constexpr int NUM_PACKS = N / VEC;
static constexpr int PACKS_PER_BLOCK = BLOCK / VEC;
static constexpr int MIDW = 18;
static constexpr int INFW = 14;
static constexpr int MID_ACC_FRAC = 16;
static constexpr int RECIP_LUT_BITS = 8;
static constexpr int RECIP_LUT_SIZE = 1 << RECIP_LUT_BITS;
static constexpr int RECIP_INDEX_SHIFT = MID_ACC_FRAC - RECIP_LUT_BITS - 1;

typedef ap_uint<32 * VEC> pack_t;
// Internal ranges are much tighter than the external 32-bit fixed-point type.
typedef ap_fixed<MIDW, 4, AP_TRN, AP_WRAP> mid_t;
typedef ap_ufixed<INFW, 2, AP_TRN, AP_WRAP> inv_t;
typedef ap_fixed<22, 6, AP_TRN, AP_WRAP> mid_acc_t;
typedef ap_ufixed<22, 10, AP_TRN, AP_WRAP> sum_abs_t;
typedef ap_uint<MIDW * VEC> mid_pack_t;
static_assert((N % VEC) == 0, "N must be divisible by VEC");
static_assert((BLOCK % VEC) == 0, "BLOCK must be divisible by VEC");

static const inv_t RECIP_LUT[RECIP_LUT_SIZE] = {
    (inv_t)1.996101364522, (inv_t)1.988349514563, (inv_t)1.980657640232, (inv_t)1.973025048170, (inv_t)1.965451055662, (inv_t)1.957934990440, (inv_t)1.950476190476, (inv_t)1.943074003795,
    (inv_t)1.935727788280, (inv_t)1.928436911488, (inv_t)1.921200750469, (inv_t)1.914018691589, (inv_t)1.906890130354, (inv_t)1.899814471243, (inv_t)1.892791127542, (inv_t)1.885819521179,
    (inv_t)1.878899082569, (inv_t)1.872029250457, (inv_t)1.865209471767, (inv_t)1.858439201452, (inv_t)1.851717902351, (inv_t)1.845045045045, (inv_t)1.838420107720, (inv_t)1.831842576029,
    (inv_t)1.825311942959, (inv_t)1.818827708703, (inv_t)1.812389380531, (inv_t)1.805996472663, (inv_t)1.799648506151, (inv_t)1.793345008757, (inv_t)1.787085514834, (inv_t)1.780869565217,
    (inv_t)1.774696707106, (inv_t)1.768566493955, (inv_t)1.762478485370, (inv_t)1.756432246998, (inv_t)1.750427350427, (inv_t)1.744463373083, (inv_t)1.738539898132, (inv_t)1.732656514382,
    (inv_t)1.726812816189, (inv_t)1.721008403361, (inv_t)1.715242881072, (inv_t)1.709515859766, (inv_t)1.703826955075, (inv_t)1.698175787728, (inv_t)1.692561983471, (inv_t)1.686985172982,
    (inv_t)1.681444991790, (inv_t)1.675941080196, (inv_t)1.670473083197, (inv_t)1.665040650407, (inv_t)1.659643435981, (inv_t)1.654281098546, (inv_t)1.648953301127, (inv_t)1.643659711075,
    (inv_t)1.638400000000, (inv_t)1.633173843700, (inv_t)1.627980922099, (inv_t)1.622820919176, (inv_t)1.617693522907, (inv_t)1.612598425197, (inv_t)1.607535321821, (inv_t)1.602503912363,
    (inv_t)1.597503900156, (inv_t)1.592534992224, (inv_t)1.587596899225, (inv_t)1.582689335394, (inv_t)1.577812018490, (inv_t)1.572964669739, (inv_t)1.568147013783, (inv_t)1.563358778626,
    (inv_t)1.558599695586, (inv_t)1.553869499241, (inv_t)1.549167927383, (inv_t)1.544494720965, (inv_t)1.539849624060, (inv_t)1.535232383808, (inv_t)1.530642750374, (inv_t)1.526080476900,
    (inv_t)1.521545319465, (inv_t)1.517037037037, (inv_t)1.512555391433, (inv_t)1.508100147275, (inv_t)1.503671071953, (inv_t)1.499267935578, (inv_t)1.494890510949, (inv_t)1.490538573508,
    (inv_t)1.486211901306, (inv_t)1.481910274964, (inv_t)1.477633477633, (inv_t)1.473381294964, (inv_t)1.469153515065, (inv_t)1.464949928469, (inv_t)1.460770328103, (inv_t)1.456614509246,
    (inv_t)1.452482269504, (inv_t)1.448373408769, (inv_t)1.444287729196, (inv_t)1.440225035162, (inv_t)1.436185133240, (inv_t)1.432167832168, (inv_t)1.428172942817, (inv_t)1.424200278164,
    (inv_t)1.420249653259, (inv_t)1.416320885201, (inv_t)1.412413793103, (inv_t)1.408528198074, (inv_t)1.404663923182, (inv_t)1.400820793434, (inv_t)1.396998635744, (inv_t)1.393197278912,
    (inv_t)1.389416553596, (inv_t)1.385656292287, (inv_t)1.381916329285, (inv_t)1.378196500673, (inv_t)1.374496644295, (inv_t)1.370816599732, (inv_t)1.367156208278, (inv_t)1.363515312916,
    (inv_t)1.359893758300, (inv_t)1.356291390728, (inv_t)1.352708058124, (inv_t)1.349143610013, (inv_t)1.345597897503, (inv_t)1.342070773263, (inv_t)1.338562091503, (inv_t)1.335071707953,
    (inv_t)1.331599479844, (inv_t)1.328145265888, (inv_t)1.324708926261, (inv_t)1.321290322581, (inv_t)1.317889317889, (inv_t)1.314505776637, (inv_t)1.311139564661, (inv_t)1.307790549170,
    (inv_t)1.304458598726, (inv_t)1.301143583227, (inv_t)1.297845373891, (inv_t)1.294563843236, (inv_t)1.291298865069, (inv_t)1.288050314465, (inv_t)1.284818067754, (inv_t)1.281602002503,
    (inv_t)1.278401997503, (inv_t)1.275217932752, (inv_t)1.272049689441, (inv_t)1.268897149938, (inv_t)1.265760197775, (inv_t)1.262638717633, (inv_t)1.259532595326, (inv_t)1.256441717791,
    (inv_t)1.253365973072, (inv_t)1.250305250305, (inv_t)1.247259439708, (inv_t)1.244228432564, (inv_t)1.241212121212, (inv_t)1.238210399033, (inv_t)1.235223160434, (inv_t)1.232250300842,
    (inv_t)1.229291716687, (inv_t)1.226347305389, (inv_t)1.223416965352, (inv_t)1.220500595948, (inv_t)1.217598097503, (inv_t)1.214709371293, (inv_t)1.211834319527, (inv_t)1.208972845336,
    (inv_t)1.206124852768, (inv_t)1.203290246769, (inv_t)1.200468933177, (inv_t)1.197660818713, (inv_t)1.194865810968, (inv_t)1.192083818393, (inv_t)1.189314750290, (inv_t)1.186558516802,
    (inv_t)1.183815028902, (inv_t)1.181084198385, (inv_t)1.178365937860, (inv_t)1.175660160735, (inv_t)1.172966781214, (inv_t)1.170285714286, (inv_t)1.167616875713, (inv_t)1.164960182025,
    (inv_t)1.162315550511, (inv_t)1.159682899207, (inv_t)1.157062146893, (inv_t)1.154453213078, (inv_t)1.151856017998, (inv_t)1.149270482604, (inv_t)1.146696528555, (inv_t)1.144134078212,
    (inv_t)1.141583054627, (inv_t)1.139043381535, (inv_t)1.136514983352, (inv_t)1.133997785161, (inv_t)1.131491712707, (inv_t)1.128996692393, (inv_t)1.126512651265, (inv_t)1.124039517014,
    (inv_t)1.121577217963, (inv_t)1.119125683060, (inv_t)1.116684841876, (inv_t)1.114254624592, (inv_t)1.111834961998, (inv_t)1.109425785482, (inv_t)1.107027027027, (inv_t)1.104638619202,
    (inv_t)1.102260495156, (inv_t)1.099892588614, (inv_t)1.097534833869, (inv_t)1.095187165775, (inv_t)1.092849519744, (inv_t)1.090521831736, (inv_t)1.088204038257, (inv_t)1.085896076352,
    (inv_t)1.083597883598, (inv_t)1.081309398099, (inv_t)1.079030558483, (inv_t)1.076761303891, (inv_t)1.074501573977, (inv_t)1.072251308901, (inv_t)1.070010449321, (inv_t)1.067778936392,
    (inv_t)1.065556711759, (inv_t)1.063343717549, (inv_t)1.061139896373, (inv_t)1.058945191313, (inv_t)1.056759545924, (inv_t)1.054582904222, (inv_t)1.052415210689, (inv_t)1.050256410256,
    (inv_t)1.048106448311, (inv_t)1.045965270684, (inv_t)1.043832823649, (inv_t)1.041709053917, (inv_t)1.039593908629, (inv_t)1.037487335360, (inv_t)1.035389282103, (inv_t)1.033299697275,
    (inv_t)1.031218529708, (inv_t)1.029145728643, (inv_t)1.027081243731, (inv_t)1.025025025025, (inv_t)1.022977022977, (inv_t)1.020937188435, (inv_t)1.018905472637, (inv_t)1.016881827210,
    (inv_t)1.014866204163, (inv_t)1.012858555885, (inv_t)1.010858835143, (inv_t)1.008866995074, (inv_t)1.006882989184, (inv_t)1.004906771344, (inv_t)1.002938295788, (inv_t)1.000977517107,
};

static inline data_t abs_fp(data_t x) {
#pragma HLS INLINE
    return (x < (data_t)0) ? (data_t)(-x) : x;
}

static inline data_t clamp_fp(data_t x, data_t lo, data_t hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline data_t unpack_lane(const pack_t &v, int lane) {
#pragma HLS INLINE
    data_t x;
    x.range(31, 0) = v.range(32 * lane + 31, 32 * lane);
    return x;
}

static inline void pack_lane(pack_t &v, int lane, const data_t &x) {
#pragma HLS INLINE
    v.range(32 * lane + 31, 32 * lane) = x.range(31, 0);
}

static inline mid_t unpack_lane(const mid_pack_t &v, int lane) {
#pragma HLS INLINE
    mid_t x;
    x.range(MIDW - 1, 0) = v.range(MIDW * lane + MIDW - 1, MIDW * lane);
    return x;
}

static inline void pack_lane(mid_pack_t &v, int lane, const mid_t &x) {
#pragma HLS INLINE
    v.range(MIDW * lane + MIDW - 1, MIDW * lane) = x.range(MIDW - 1, 0);
}

static inline mid_t preprocess_coeff(data_t x) {
#pragma HLS INLINE
    mid_acc_t ax = (mid_acc_t)x;
    return (mid_t)(ax - (ax >> 3) + (mid_acc_t)0.125);
}

static inline mid_t transform_core(mid_t x0, mid_t x1, mid_t x2) {
#pragma HLS INLINE
    mid_acc_t a0 = (mid_acc_t)x0;
    mid_acc_t a1 = (mid_acc_t)x1;
    mid_acc_t a2 = (mid_acc_t)x2;
    return (mid_t)((a0 >> 1) - (a1 >> 2) + (a2 >> 3));
}

static inline data_t postprocess_coeff(mid_t x) {
#pragma HLS INLINE
    mid_acc_t ax = (mid_acc_t)x;
    return (data_t)(ax + (ax >> 2) + (mid_acc_t)0.05);
}

static inline mid_t normalize_coeff(mid_t x, inv_t inv) {
#pragma HLS INLINE
    mid_acc_t prod;
#pragma HLS BIND_OP variable=prod op=mul impl=DSP latency=2
    prod = (mid_acc_t)x * (mid_acc_t)inv;
    return (mid_t)prod;
}

static inline inv_t reciprocal_lut_unit(mid_acc_t xn) {
#pragma HLS INLINE
    mid_acc_t x_clamped = xn;
    if (x_clamped < (mid_acc_t)0.5) x_clamped = (mid_acc_t)0.5;
    if (x_clamped >= (mid_acc_t)1.0) x_clamped = (mid_acc_t)0.9999847412109375;

    ap_ufixed<MID_ACC_FRAC, 0, AP_TRN, AP_WRAP> delta =
        (ap_ufixed<MID_ACC_FRAC, 0, AP_TRN, AP_WRAP>)(x_clamped - (mid_acc_t)0.5);
    ap_uint<MID_ACC_FRAC> raw = delta.range(MID_ACC_FRAC - 1, 0);
    ap_uint<RECIP_LUT_BITS> idx = (ap_uint<RECIP_LUT_BITS>)(raw >> RECIP_INDEX_SHIFT);
    return RECIP_LUT[idx];
}

static inline inv_t reciprocal_nr(stat_t x) {
#pragma HLS INLINE
    mid_acc_t xn = (mid_acc_t)x;

    // Normalize into [0.5, 1) with shifts, then use a small LUT-based
    // reciprocal. This removes the Newton-Raphson DSP chain from the
    // block-statistics stage.
    if (xn >= (mid_acc_t)16.0) {
        mid_acc_t y = (mid_acc_t)reciprocal_lut_unit(xn >> 5);
        return (inv_t)(y >> 5);
    }
    if (xn >= (mid_acc_t)8.0) {
        mid_acc_t y = (mid_acc_t)reciprocal_lut_unit(xn >> 4);
        return (inv_t)(y >> 4);
    }
    if (xn >= (mid_acc_t)4.0) {
        mid_acc_t y = (mid_acc_t)reciprocal_lut_unit(xn >> 3);
        return (inv_t)(y >> 3);
    }
    if (xn >= (mid_acc_t)2.0) {
        mid_acc_t y = (mid_acc_t)reciprocal_lut_unit(xn >> 2);
        return (inv_t)(y >> 2);
    }
    if (xn >= (mid_acc_t)1.0) {
        mid_acc_t y = (mid_acc_t)reciprocal_lut_unit(xn >> 1);
        return (inv_t)(y >> 1);
    }
    return reciprocal_lut_unit(xn);
}

static void preprocess_stage(const data_t in[N],
                             hls::stream<mid_pack_t> &out_s0) {
#pragma HLS INLINE off

#ifdef __SYNTHESIS__
    const pack_t *in_wide = reinterpret_cast<const pack_t *>(&in[0]);
#endif

    for (int p = 0; p < NUM_PACKS; p++) {
#pragma HLS PIPELINE II=1
        mid_pack_t s0_pack = 0;

#ifdef __SYNTHESIS__
        pack_t in_pack = in_wide[p];
#else
        pack_t in_pack = 0;
        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            pack_lane(in_pack, lane, in[p * VEC + lane]);
        }
#endif

        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            data_t x = unpack_lane(in_pack, lane);
            mid_t s0_lane = preprocess_coeff(x);
            pack_lane(s0_pack, lane, s0_lane);
        }

        out_s0.write(s0_pack);
    }
}

static void transform_and_invstats(hls::stream<mid_pack_t> &in_s0,
                                   hls::stream<mid_pack_t> &out_s1,
                                   hls::stream<inv_t> &inv_stats_s) {
#pragma HLS INLINE off
    const stat_t eps = (stat_t)0.5;

    mid_t prev1 = 0;
    mid_t prev2 = 0;
    sum_abs_t sum_abs = 0;

    for (int p = 0; p < NUM_PACKS; p++) {
#pragma HLS PIPELINE II=1
        mid_pack_t out_pack = 0;
        mid_t s0[VEC];
#pragma HLS ARRAY_PARTITION variable=s0 complete

        mid_pack_t in_pack = in_s0.read();

        sum_abs_t lane_sum = 0;
        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            mid_t s0_lane = unpack_lane(in_pack, lane);
            s0[lane] = s0_lane;
            lane_sum += (s0_lane < (mid_t)0) ? (sum_abs_t)(-s0_lane)
                                             : (sum_abs_t)s0_lane;
        }

        sum_abs += lane_sum;

        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            mid_t x0 = s0[lane];
            mid_t x1 = (lane == 0) ? prev1 : s0[lane - 1];
            mid_t x2 = (lane == 0) ? prev2 : ((lane == 1) ? prev1 : s0[lane - 2]);

            mid_t y = transform_core(x0, x1, x2);
            y = (y < (mid_t)0) ? (mid_t)(-y) : y;
            if (y > (mid_t)7.5) y = (mid_t)7.5;
            pack_lane(out_pack, lane, y);
        }

        prev2 = s0[VEC - 2];
        prev1 = s0[VEC - 1];
        out_s1.write(out_pack);

        if ((p % PACKS_PER_BLOCK) == (PACKS_PER_BLOCK - 1)) {
            stat_t avg_abs = (stat_t)(sum_abs / (sum_abs_t)BLOCK);
            inv_stats_s.write(reciprocal_nr(avg_abs + eps));
            sum_abs = 0;
        }
    }
}

static void join_and_normalize(hls::stream<mid_pack_t> &in_s1,
                               hls::stream<inv_t> &inv_stats_s,
                               hls::stream<mid_pack_t> &out_s3) {
#pragma HLS INLINE off

    // Ping-pong block buffers let the stage consume the next block while
    // normalizing the current block, cutting the block cost from
    // read-then-write to effectively one pack per cycle.
    mid_pack_t block_buf0[PACKS_PER_BLOCK];
    mid_pack_t block_buf1[PACKS_PER_BLOCK];
#pragma HLS BIND_STORAGE variable=block_buf0 type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=block_buf1 type=ram_s2p impl=lutram

    for (int p = 0; p < PACKS_PER_BLOCK; p++) {
#pragma HLS PIPELINE II=1
        block_buf0[p] = in_s1.read();
    }

    inv_t inv_curr = inv_stats_s.read();
    bool curr_is_buf0 = true;

    for (int b = 0; b < NUM_BLOCKS - 1; b++) {
        for (int p = 0; p < PACKS_PER_BLOCK; p++) {
#pragma HLS PIPELINE II=1
            mid_pack_t next_pack = in_s1.read();
            mid_pack_t in_pack = curr_is_buf0 ? block_buf0[p] : block_buf1[p];
            if (curr_is_buf0) {
                block_buf1[p] = next_pack;
            } else {
                block_buf0[p] = next_pack;
            }

            mid_pack_t out_pack = 0;
            for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
                mid_t x = unpack_lane(in_pack, lane);
                mid_t n = normalize_coeff(x, inv_curr);
                pack_lane(out_pack, lane, n);
            }

            out_s3.write(out_pack);
        }

        inv_curr = inv_stats_s.read();
        curr_is_buf0 = !curr_is_buf0;
    }

    for (int p = 0; p < PACKS_PER_BLOCK; p++) {
#pragma HLS PIPELINE II=1
        mid_pack_t in_pack = curr_is_buf0 ? block_buf0[p] : block_buf1[p];
        mid_pack_t out_pack = 0;
        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            mid_t x = unpack_lane(in_pack, lane);
            mid_t n = normalize_coeff(x, inv_curr);
            pack_lane(out_pack, lane, n);
        }

        out_s3.write(out_pack);
    }
}

static void postprocess_and_store(hls::stream<mid_pack_t> &in_s3,
                                  data_t out[N]) {
#pragma HLS INLINE off

#ifdef __SYNTHESIS__
    pack_t *out_wide = reinterpret_cast<pack_t *>(&out[0]);
#endif

    for (int p = 0; p < NUM_PACKS; p++) {
#pragma HLS PIPELINE II=1
        mid_pack_t in_pack = in_s3.read();
        pack_t out_pack = 0;

        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            mid_t x = unpack_lane(in_pack, lane);
            data_t z = postprocess_coeff(x);
            z = clamp_fp(z, (data_t)0, (data_t)7.9);
            pack_lane(out_pack, lane, z);
        }

#ifdef __SYNTHESIS__
        out_wide[p] = out_pack;
#else
        for (int lane = 0; lane < VEC; lane++) {
#pragma HLS UNROLL
            out[p * VEC + lane] = unpack_lane(out_pack, lane);
        }
#endif
    }
}

void top_kernel(const data_t in[N], data_t out[N]) {
#pragma HLS interface m_axi port=in offset=slave bundle=in max_widen_bitwidth=1024 max_read_burst_length=64 num_read_outstanding=2
#pragma HLS interface m_axi port=out offset=slave bundle=out max_widen_bitwidth=1024 max_write_burst_length=64 num_write_outstanding=2
#pragma HLS interface s_axilite port=return

#pragma HLS DATAFLOW

    hls::stream<mid_pack_t> s0_stream("s0_stream");
    hls::stream<mid_pack_t> s1_stream("s1_stream");
    hls::stream<inv_t> inv_stat_stream("inv_stat_stream");
    hls::stream<mid_pack_t> s3_stream("s3_stream");

#pragma HLS STREAM variable=s0_stream depth=4
#pragma HLS STREAM variable=s1_stream depth=16
#pragma HLS STREAM variable=inv_stat_stream depth=8
#pragma HLS STREAM variable=s3_stream depth=2
#pragma HLS BIND_STORAGE variable=s3_stream type=fifo impl=srl

    preprocess_stage(in, s0_stream);
    transform_and_invstats(s0_stream, s1_stream, inv_stat_stream);
    join_and_normalize(s1_stream, inv_stat_stream, s3_stream);
    postprocess_and_store(s3_stream, out);
}
