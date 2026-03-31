#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

// 1024-bit raw AXI type for doubled bandwidth (32 elements per cycle)
typedef ap_uint<1024> ultra_wide_t;

// Narrow internal types (smaller than dcl.h types, used only inside top.cpp)
typedef ap_fixed<16, 4, AP_TRN, AP_WRAP> narrow_t;    // 16-bit internal data (vs 32-bit data_t)
typedef ap_fixed<16, 3, AP_TRN, AP_WRAP> nstat_t;     // 16-bit internal stat (vs 32-bit stat_t)

// The SIMD Vector type for streaming between kernels (32 x 16-bit = 512 bits)
struct vec32_t {
    narrow_t data[32];
};

#define BLOCK_INV ((acc_t)(1.0 / BLOCK))
const int VEC_SIZE = 32;
const int CHUNKS = N / VEC_SIZE;           // 2048
const int CHUNKS_PER_BLOCK = BLOCK / VEC_SIZE; // 8

// K3 Div
// scale replaced with shift-add in K3_fused: 1024/3.5 ≈ 292.5 = 256+32+4+0.5
const int NUM_BLOCKS = N / BLOCK;

// Narrow types for DSP48-friendly multiply
typedef ap_fixed<16, 2, AP_TRN, AP_WRAP> inv_narrow_t;   // reciprocal LUT values [0.25..2]
typedef ap_fixed<16, 4, AP_TRN, AP_WRAP> data_narrow_t;  // K3 mult input [0..7.5]

// Narrow accumulator (22-bit, 12 frac bits for 0.05% resolution)
typedef ap_fixed<22, 10, AP_TRN, AP_WRAP> nacc_t;

// Narrow FIR intermediate type (14-bit, range [-8, 8))
typedef ap_fixed<14, 4, AP_TRN, AP_WRAP> fir_t;

// ============================================================================
// Helper Functions
// ============================================================================

static inline narrow_t abs_n(narrow_t x) {
    #pragma HLS INLINE
    return (x < (narrow_t)0) ? (narrow_t)(-x) : x;
}

static inline narrow_t clamp_n(narrow_t x, narrow_t lo, narrow_t hi) {
    #pragma HLS INLINE
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

// ============================================================================
// 1. AXI Read with Dual Output (replaces axi_read + K0 split)
// ============================================================================

void axi_read_split(const data_t in[N],
                    hls::stream<vec32_t>& raw_K1,
                    hls::stream<vec32_t>& raw_K2) {
    #pragma HLS INLINE off
    const ultra_wide_t *flat_in = (const ultra_wide_t*)in;
    
    for (int k = 0; k < CHUNKS; k++) {
        #pragma HLS PIPELINE II=1
        ultra_wide_t raw_chunk = flat_in[k];
        vec32_t vec;
        #pragma HLS ARRAY_PARTITION variable=vec.data complete
        
        for (int p = 0; p < VEC_SIZE; p++) {
            #pragma HLS UNROLL
            unsigned int raw32 = raw_chunk.range(31 + 32*p, 32*p);
            vec.data[p] = *(data_t*)(&raw32);
        }
        raw_K1.write(vec);
        raw_K2.write(vec);
    }
}

// ============================================================================
// 2. K1_fused: K0 preprocess + K1 transform (fused)
// ============================================================================

void K1_fused(hls::stream<vec32_t>& raw_stream, hls::stream<vec32_t>& s1_stream) {
    #pragma HLS INLINE off
    // K1 FIR coefficients: w0=0.5, w1=-0.25, w2=0.125 → all powers of 2 → use shifts
    
    // State: last two K0-preprocessed elements from the previous chunk
    // Non-static to enable rewind (reset to 0 each invocation)
    fir_t prev_x1 = 0;
    fir_t prev_x2 = 0;
    
    for (int k = 0; k < CHUNKS; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LATENCY min=9 max=12
        vec32_t in_vec = raw_stream.read();
        vec32_t out_vec;
        #pragma HLS ARRAY_PARTITION variable=out_vec.data complete
        
        // Single merged loop: K0 preprocess (shift-subtract) + K1 FIR filter (shifts)
        for (int p = 0; p < VEC_SIZE; p++) {
            #pragma HLS UNROLL
            // K0: 0.875 * x + 0.125 = x - (x >> 3) + 0.125
            fir_t raw0 = (fir_t)in_vec.data[p];
            fir_t x0 = (fir_t)(raw0 - (raw0 >> 3) + (fir_t)0.125);
            
            fir_t x1;
            if (p >= 1) {
                fir_t raw1 = (fir_t)in_vec.data[p-1];
                x1 = (fir_t)(raw1 - (raw1 >> 3) + (fir_t)0.125);
            } else {
                x1 = (fir_t)prev_x1;
            }
            
            fir_t x2;
            if (p >= 2) {
                fir_t raw2 = (fir_t)in_vec.data[p-2];
                x2 = (fir_t)(raw2 - (raw2 >> 3) + (fir_t)0.125);
            } else if (p == 1) {
                x2 = (fir_t)prev_x1;
            } else {
                x2 = (fir_t)prev_x2;
            }

            // K1 FIR: 0.5*x0 - 0.25*x1 + 0.125*x2 → shifts only
            fir_t m0 = x0 >> 1;        // 0.5 * x0
            fir_t m1 = -(x1 >> 2);     // -0.25 * x1
            fir_t m2 = x2 >> 3;        // 0.125 * x2
            fir_t sum01 = m0 + m1;
            fir_t fir_acc = sum01 + m2;
            out_vec.data[p] = clamp_n(abs_n((narrow_t)fir_acc), (narrow_t)0, (narrow_t)7.5);
        }
        
        // Save last two preprocessed elements for next chunk
        fir_t r31 = (fir_t)in_vec.data[31];
        prev_x1 = (fir_t)(r31 - (r31 >> 3) + (fir_t)0.125);
        fir_t r30 = (fir_t)in_vec.data[30];
        prev_x2 = (fir_t)(r30 - (r30 >> 3) + (fir_t)0.125);
        
        s1_stream.write(out_vec);
    }
}

// ============================================================================
// 3. K2_fused: K0 preprocess + K2 block stat (fused)
// ============================================================================

void K2_fused(hls::stream<vec32_t>& raw_stream, hls::stream<nstat_t>& stat_stream) {
    #pragma HLS INLINE off
    // K2 constants
    const nstat_t eps = (nstat_t)0.5;

    nacc_t block_chunks[CHUNKS_PER_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=block_chunks complete

    for (int i = 0; i < CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LATENCY min=14 max=18
        
        vec32_t in_vec = raw_stream.read();
        
        // Step 1: K0 preprocess + abs for all 32 elements (narrow 20-bit)
        fir_t vals[32];
        #pragma HLS ARRAY_PARTITION variable=vals complete
        for (int p = 0; p < VEC_SIZE; p++) {
            #pragma HLS UNROLL
            fir_t raw = (fir_t)in_vec.data[p];
            fir_t preprocessed = (fir_t)(raw - (raw >> 3) + (fir_t)0.125);
            vals[p] = (preprocessed < 0) ? (fir_t)(-preprocessed) : preprocessed;
        }
        
        // Step 2: Balanced binary adder tree (5 levels instead of 31 serial adds)
        // Level 1: 32 → 16 (narrow 20-bit)
        fir_t l1[16];
        #pragma HLS ARRAY_PARTITION variable=l1 complete
        for (int j = 0; j < 16; j++) {
            #pragma HLS UNROLL
            l1[j] = vals[2*j] + vals[2*j+1];
        }
        // Level 2: 16 → 8 (widen to 32-bit for accumulated sums)
        nacc_t l2[8];
        #pragma HLS ARRAY_PARTITION variable=l2 complete
        for (int j = 0; j < 8; j++) {
            #pragma HLS UNROLL
            l2[j] = (nacc_t)l1[2*j] + (nacc_t)l1[2*j+1];
        }
        // Level 3: 8 → 4
        nacc_t l3_0 = l2[0] + l2[1];
        nacc_t l3_1 = l2[2] + l2[3];
        nacc_t l3_2 = l2[4] + l2[5];
        nacc_t l3_3 = l2[6] + l2[7];
        // Level 4: 4 → 2
        nacc_t l4_0 = l3_0 + l3_1;
        nacc_t l4_1 = l3_2 + l3_3;
        // Level 5: 2 → 1
        nacc_t chunk_sum = l4_0 + l4_1;
        
        int slot = i & (CHUNKS_PER_BLOCK - 1);
        block_chunks[slot] = chunk_sum;
        
        if (slot == CHUNKS_PER_BLOCK - 1) {
            nacc_t s01 = block_chunks[0] + block_chunks[1];
            nacc_t s23 = block_chunks[2] + block_chunks[3];
            nacc_t s45 = block_chunks[4] + block_chunks[5];
            nacc_t s67 = block_chunks[6] + chunk_sum;
            nacc_t s0123 = s01 + s23;
            nacc_t s4567 = s45 + s67;
            nacc_t total = s0123 + s4567;
            stat_stream.write((nstat_t)(total >> 8) + eps);
        }
    }
}

// ============================================================================
// 4. K3_fused: K3_div (LUT) + K3_mult + K4_postprocess (all fused)
// ============================================================================

void K3_fused(hls::stream<vec32_t>& s1_stream,
              hls::stream<nstat_t>& stat_stream,
              hls::stream<vec32_t>& out_stream) {
    #pragma HLS INLINE off

    // BRAM-based reciprocal LUT: 1024 entries for x in [0.5, 4.0)
    static const int LUT_SIZE = 1024;
    static const nstat_t inv_lut[LUT_SIZE] = {
    (stat_t)2.00000000, (stat_t)1.98642095, (stat_t)1.97302505, (stat_t)1.95980861, (stat_t)1.94676806, (stat_t)1.93389991, (stat_t)1.92120075, (stat_t)1.90866729,
    (stat_t)1.89629630, (stat_t)1.88408464, (stat_t)1.87202925, (stat_t)1.86012716, (stat_t)1.84837545, (stat_t)1.83677130, (stat_t)1.82531194, (stat_t)1.81399469,
    (stat_t)1.80281690, (stat_t)1.79177603, (stat_t)1.78086957, (stat_t)1.77009507, (stat_t)1.75945017, (stat_t)1.74893254, (stat_t)1.73853990, (stat_t)1.72827004,
    (stat_t)1.71812081, (stat_t)1.70809008, (stat_t)1.69817579, (stat_t)1.68837593, (stat_t)1.67868852, (stat_t)1.66911165, (stat_t)1.65964344, (stat_t)1.65028203,
    (stat_t)1.64102564, (stat_t)1.63187251, (stat_t)1.62282092, (stat_t)1.61386919, (stat_t)1.60501567, (stat_t)1.59625877, (stat_t)1.58759690, (stat_t)1.57902853,
    (stat_t)1.57055215, (stat_t)1.56216629, (stat_t)1.55386950, (stat_t)1.54566038, (stat_t)1.53753754, (stat_t)1.52949963, (stat_t)1.52154532, (stat_t)1.51367332,
    (stat_t)1.50588235, (stat_t)1.49817118, (stat_t)1.49053857, (stat_t)1.48298335, (stat_t)1.47550432, (stat_t)1.46810036, (stat_t)1.46077033, (stat_t)1.45351313,
    (stat_t)1.44632768, (stat_t)1.43921293, (stat_t)1.43216783, (stat_t)1.42519137, (stat_t)1.41828255, (stat_t)1.41144039, (stat_t)1.40466392, (stat_t)1.39795222,
    (stat_t)1.39130435, (stat_t)1.38471941, (stat_t)1.37819650, (stat_t)1.37173476, (stat_t)1.36533333, (stat_t)1.35899137, (stat_t)1.35270806, (stat_t)1.34648258,
    (stat_t)1.34031414, (stat_t)1.33420195, (stat_t)1.32814527, (stat_t)1.32214332, (stat_t)1.31619537, (stat_t)1.31030070, (stat_t)1.30445860, (stat_t)1.29866836,
    (stat_t)1.29292929, (stat_t)1.28724073, (stat_t)1.28160200, (stat_t)1.27601246, (stat_t)1.27047146, (stat_t)1.26497838, (stat_t)1.25953260, (stat_t)1.25413350,
    (stat_t)1.24878049, (stat_t)1.24347298, (stat_t)1.23821040, (stat_t)1.23299217, (stat_t)1.22781775, (stat_t)1.22268657, (stat_t)1.21759810, (stat_t)1.21255181,
    (stat_t)1.20754717, (stat_t)1.20258368, (stat_t)1.19766082, (stat_t)1.19277810, (stat_t)1.18793503, (stat_t)1.18313114, (stat_t)1.17836594, (stat_t)1.17363897,
    (stat_t)1.16894977, (stat_t)1.16429790, (stat_t)1.15968290, (stat_t)1.15510434, (stat_t)1.15056180, (stat_t)1.14605484, (stat_t)1.14158305, (stat_t)1.13714603,
    (stat_t)1.13274336, (stat_t)1.12837466, (stat_t)1.12403952, (stat_t)1.11973756, (stat_t)1.11546841, (stat_t)1.11123169, (stat_t)1.10702703, (stat_t)1.10285407,
    (stat_t)1.09871245, (stat_t)1.09460182, (stat_t)1.09052183, (stat_t)1.08647215, (stat_t)1.08245243, (stat_t)1.07846235, (stat_t)1.07450157, (stat_t)1.07056979,
    (stat_t)1.06666667, (stat_t)1.06279190, (stat_t)1.05894519, (stat_t)1.05512622, (stat_t)1.05133470, (stat_t)1.04757033, (stat_t)1.04383282, (stat_t)1.04012189,
    (stat_t)1.03643725, (stat_t)1.03277862, (stat_t)1.02914573, (stat_t)1.02553831, (stat_t)1.02195609, (stat_t)1.01839881, (stat_t)1.01486620, (stat_t)1.01135802,
    (stat_t)1.00787402, (stat_t)1.00441393, (stat_t)1.00097752, (stat_t)0.99756454, (stat_t)0.99417476, (stat_t)0.99080793, (stat_t)0.98746384, (stat_t)0.98414224,
    (stat_t)0.98084291, (stat_t)0.97756563, (stat_t)0.97431018, (stat_t)0.97107634, (stat_t)0.96786389, (stat_t)0.96467263, (stat_t)0.96150235, (stat_t)0.95835283,
    (stat_t)0.95522388, (stat_t)0.95211530, (stat_t)0.94902688, (stat_t)0.94595843, (stat_t)0.94290976, (stat_t)0.93988068, (stat_t)0.93687100, (stat_t)0.93388053,
    (stat_t)0.93090909, (stat_t)0.92795650, (stat_t)0.92502258, (stat_t)0.92210716, (stat_t)0.91921005, (stat_t)0.91633110, (stat_t)0.91347012, (stat_t)0.91062695,
    (stat_t)0.90780142, (stat_t)0.90499337, (stat_t)0.90220264, (stat_t)0.89942907, (stat_t)0.89667250, (stat_t)0.89393278, (stat_t)0.89120975, (stat_t)0.88850325,
    (stat_t)0.88581315, (stat_t)0.88313928, (stat_t)0.88048151, (stat_t)0.87783969, (stat_t)0.87521368, (stat_t)0.87260332, (stat_t)0.87000850, (stat_t)0.86742906,
    (stat_t)0.86486486, (stat_t)0.86231579, (stat_t)0.85978170, (stat_t)0.85726245, (stat_t)0.85475793, (stat_t)0.85226800, (stat_t)0.84979253, (stat_t)0.84733140,
    (stat_t)0.84488449, (stat_t)0.84245167, (stat_t)0.84003281, (stat_t)0.83762781, (stat_t)0.83523654, (stat_t)0.83285889, (stat_t)0.83049473, (stat_t)0.82814395,
    (stat_t)0.82580645, (stat_t)0.82348211, (stat_t)0.82117081, (stat_t)0.81887245, (stat_t)0.81658692, (stat_t)0.81431412, (stat_t)0.81205393, (stat_t)0.80980625,
    (stat_t)0.80757098, (stat_t)0.80534801, (stat_t)0.80313725, (stat_t)0.80093860, (stat_t)0.79875195, (stat_t)0.79657721, (stat_t)0.79441427, (stat_t)0.79226306,
    (stat_t)0.79012346, (stat_t)0.78799538, (stat_t)0.78587874, (stat_t)0.78377344, (stat_t)0.78167939, (stat_t)0.77959650, (stat_t)0.77752468, (stat_t)0.77546384,
    (stat_t)0.77341390, (stat_t)0.77137476, (stat_t)0.76934636, (stat_t)0.76732859, (stat_t)0.76532138, (stat_t)0.76332464, (stat_t)0.76133829, (stat_t)0.75936225,
    (stat_t)0.75739645, (stat_t)0.75544080, (stat_t)0.75349522, (stat_t)0.75155963, (stat_t)0.74963397, (stat_t)0.74771815, (stat_t)0.74581209, (stat_t)0.74391573,
    (stat_t)0.74202899, (stat_t)0.74015179, (stat_t)0.73828407, (stat_t)0.73642575, (stat_t)0.73457676, (stat_t)0.73273703, (stat_t)0.73090650, (stat_t)0.72908508,
    (stat_t)0.72727273, (stat_t)0.72546936, (stat_t)0.72367491, (stat_t)0.72188932, (stat_t)0.72011252, (stat_t)0.71834444, (stat_t)0.71658502, (stat_t)0.71483421,
    (stat_t)0.71309192, (stat_t)0.71135811, (stat_t)0.70963271, (stat_t)0.70791566, (stat_t)0.70620690, (stat_t)0.70450636, (stat_t)0.70281400, (stat_t)0.70112975,
    (stat_t)0.69945355, (stat_t)0.69778535, (stat_t)0.69612508, (stat_t)0.69447270, (stat_t)0.69282815, (stat_t)0.69119136, (stat_t)0.68956229, (stat_t)0.68794088,
    (stat_t)0.68632708, (stat_t)0.68472083, (stat_t)0.68312208, (stat_t)0.68153078, (stat_t)0.67994688, (stat_t)0.67837032, (stat_t)0.67680106, (stat_t)0.67523904,
    (stat_t)0.67368421, (stat_t)0.67213653, (stat_t)0.67059594, (stat_t)0.66906240, (stat_t)0.66753585, (stat_t)0.66601626, (stat_t)0.66450357, (stat_t)0.66299773,
    (stat_t)0.66149871, (stat_t)0.66000645, (stat_t)0.65852090, (stat_t)0.65704203, (stat_t)0.65556978, (stat_t)0.65410412, (stat_t)0.65264500, (stat_t)0.65119237,
    (stat_t)0.64974619, (stat_t)0.64830643, (stat_t)0.64687303, (stat_t)0.64544595, (stat_t)0.64402516, (stat_t)0.64261061, (stat_t)0.64120225, (stat_t)0.63980006,
    (stat_t)0.63840399, (stat_t)0.63701400, (stat_t)0.63563004, (stat_t)0.63425209, (stat_t)0.63288010, (stat_t)0.63151403, (stat_t)0.63015385, (stat_t)0.62879951,
    (stat_t)0.62745098, (stat_t)0.62610822, (stat_t)0.62477120, (stat_t)0.62343988, (stat_t)0.62211422, (stat_t)0.62079418, (stat_t)0.61947973, (stat_t)0.61817084,
    (stat_t)0.61686747, (stat_t)0.61556958, (stat_t)0.61427714, (stat_t)0.61299012, (stat_t)0.61170848, (stat_t)0.61043219, (stat_t)0.60916121, (stat_t)0.60789552,
    (stat_t)0.60663507, (stat_t)0.60537984, (stat_t)0.60412979, (stat_t)0.60288490, (stat_t)0.60164512, (stat_t)0.60041044, (stat_t)0.59918081, (stat_t)0.59795620,
    (stat_t)0.59673660, (stat_t)0.59552195, (stat_t)0.59431225, (stat_t)0.59310744, (stat_t)0.59190751, (stat_t)0.59071243, (stat_t)0.58952216, (stat_t)0.58833668,
    (stat_t)0.58715596, (stat_t)0.58597997, (stat_t)0.58480868, (stat_t)0.58364206, (stat_t)0.58248009, (stat_t)0.58132274, (stat_t)0.58016997, (stat_t)0.57902177,
    (stat_t)0.57787810, (stat_t)0.57673895, (stat_t)0.57560427, (stat_t)0.57447405, (stat_t)0.57334826, (stat_t)0.57222688, (stat_t)0.57110987, (stat_t)0.56999722,
    (stat_t)0.56888889, (stat_t)0.56778486, (stat_t)0.56668511, (stat_t)0.56558962, (stat_t)0.56449835, (stat_t)0.56341128, (stat_t)0.56232839, (stat_t)0.56124966,
    (stat_t)0.56017505, (stat_t)0.55910456, (stat_t)0.55803815, (stat_t)0.55697580, (stat_t)0.55591748, (stat_t)0.55486318, (stat_t)0.55381287, (stat_t)0.55276653,
    (stat_t)0.55172414, (stat_t)0.55068567, (stat_t)0.54965110, (stat_t)0.54862041, (stat_t)0.54759358, (stat_t)0.54657059, (stat_t)0.54555141, (stat_t)0.54453603,
    (stat_t)0.54352442, (stat_t)0.54251656, (stat_t)0.54151243, (stat_t)0.54051201, (stat_t)0.53951528, (stat_t)0.53852222, (stat_t)0.53753281, (stat_t)0.53654703,
    (stat_t)0.53556485, (stat_t)0.53458627, (stat_t)0.53361126, (stat_t)0.53263979, (stat_t)0.53167186, (stat_t)0.53070744, (stat_t)0.52974651, (stat_t)0.52878905,
    (stat_t)0.52783505, (stat_t)0.52688449, (stat_t)0.52593734, (stat_t)0.52499359, (stat_t)0.52405322, (stat_t)0.52311622, (stat_t)0.52218256, (stat_t)0.52125223,
    (stat_t)0.52032520, (stat_t)0.51940147, (stat_t)0.51848101, (stat_t)0.51756381, (stat_t)0.51664985, (stat_t)0.51573911, (stat_t)0.51483157, (stat_t)0.51392723,
    (stat_t)0.51302605, (stat_t)0.51212803, (stat_t)0.51123315, (stat_t)0.51034139, (stat_t)0.50945274, (stat_t)0.50856717, (stat_t)0.50768468, (stat_t)0.50680525,
    (stat_t)0.50592885, (stat_t)0.50505549, (stat_t)0.50418513, (stat_t)0.50331777, (stat_t)0.50245339, (stat_t)0.50159197, (stat_t)0.50073350, (stat_t)0.49987796,
    (stat_t)0.49902534, (stat_t)0.49817563, (stat_t)0.49732880, (stat_t)0.49648485, (stat_t)0.49564376, (stat_t)0.49480551, (stat_t)0.49397009, (stat_t)0.49313749,
    (stat_t)0.49230769, (stat_t)0.49148068, (stat_t)0.49065644, (stat_t)0.48983497, (stat_t)0.48901624, (stat_t)0.48820024, (stat_t)0.48738696, (stat_t)0.48657638,
    (stat_t)0.48576850, (stat_t)0.48496330, (stat_t)0.48416076, (stat_t)0.48336087, (stat_t)0.48256362, (stat_t)0.48176900, (stat_t)0.48097698, (stat_t)0.48018757,
    (stat_t)0.47940075, (stat_t)0.47861650, (stat_t)0.47783481, (stat_t)0.47705567, (stat_t)0.47627907, (stat_t)0.47550499, (stat_t)0.47473343, (stat_t)0.47396436,
    (stat_t)0.47319778, (stat_t)0.47243368, (stat_t)0.47167204, (stat_t)0.47091285, (stat_t)0.47015611, (stat_t)0.46940179, (stat_t)0.46864989, (stat_t)0.46790039,
    (stat_t)0.46715328, (stat_t)0.46640856, (stat_t)0.46566621, (stat_t)0.46492622, (stat_t)0.46418858, (stat_t)0.46345327, (stat_t)0.46272029, (stat_t)0.46198962,
    (stat_t)0.46126126, (stat_t)0.46053519, (stat_t)0.45981141, (stat_t)0.45908989, (stat_t)0.45837064, (stat_t)0.45765363, (stat_t)0.45693887, (stat_t)0.45622633,
    (stat_t)0.45551601, (stat_t)0.45480791, (stat_t)0.45410200, (stat_t)0.45339827, (stat_t)0.45269673, (stat_t)0.45199735, (stat_t)0.45130013, (stat_t)0.45060506,
    (stat_t)0.44991213, (stat_t)0.44922132, (stat_t)0.44853263, (stat_t)0.44784605, (stat_t)0.44716157, (stat_t)0.44647918, (stat_t)0.44579887, (stat_t)0.44512063,
    (stat_t)0.44444444, (stat_t)0.44377031, (stat_t)0.44309823, (stat_t)0.44242817, (stat_t)0.44176014, (stat_t)0.44109412, (stat_t)0.44043011, (stat_t)0.43976809,
    (stat_t)0.43910806, (stat_t)0.43845001, (stat_t)0.43779393, (stat_t)0.43713981, (stat_t)0.43648764, (stat_t)0.43583741, (stat_t)0.43518912, (stat_t)0.43454275,
    (stat_t)0.43389831, (stat_t)0.43325576, (stat_t)0.43261512, (stat_t)0.43197638, (stat_t)0.43133951, (stat_t)0.43070452, (stat_t)0.43007140, (stat_t)0.42944013,
    (stat_t)0.42881072, (stat_t)0.42818315, (stat_t)0.42755741, (stat_t)0.42693350, (stat_t)0.42631141, (stat_t)0.42569112, (stat_t)0.42507264, (stat_t)0.42445596,
    (stat_t)0.42384106, (stat_t)0.42322794, (stat_t)0.42261659, (stat_t)0.42200701, (stat_t)0.42139918, (stat_t)0.42079310, (stat_t)0.42018876, (stat_t)0.41958615,
    (stat_t)0.41898527, (stat_t)0.41838611, (stat_t)0.41778866, (stat_t)0.41719291, (stat_t)0.41659886, (stat_t)0.41600650, (stat_t)0.41541582, (stat_t)0.41482682,
    (stat_t)0.41423948, (stat_t)0.41365381, (stat_t)0.41306979, (stat_t)0.41248741, (stat_t)0.41190668, (stat_t)0.41132758, (stat_t)0.41075010, (stat_t)0.41017424,
    (stat_t)0.40960000, (stat_t)0.40902736, (stat_t)0.40845632, (stat_t)0.40788688, (stat_t)0.40731901, (stat_t)0.40675273, (stat_t)0.40618802, (stat_t)0.40562488,
    (stat_t)0.40506329, (stat_t)0.40450326, (stat_t)0.40394477, (stat_t)0.40338783, (stat_t)0.40283242, (stat_t)0.40227853, (stat_t)0.40172617, (stat_t)0.40117532,
    (stat_t)0.40062598, (stat_t)0.40007814, (stat_t)0.39953180, (stat_t)0.39898695, (stat_t)0.39844358, (stat_t)0.39790169, (stat_t)0.39736127, (stat_t)0.39682232,
    (stat_t)0.39628483, (stat_t)0.39574879, (stat_t)0.39521420, (stat_t)0.39468106, (stat_t)0.39414935, (stat_t)0.39361907, (stat_t)0.39309021, (stat_t)0.39256278,
    (stat_t)0.39203675, (stat_t)0.39151214, (stat_t)0.39098893, (stat_t)0.39046711, (stat_t)0.38994669, (stat_t)0.38942765, (stat_t)0.38890999, (stat_t)0.38839370,
    (stat_t)0.38787879, (stat_t)0.38736524, (stat_t)0.38685304, (stat_t)0.38634220, (stat_t)0.38583271, (stat_t)0.38532455, (stat_t)0.38481774, (stat_t)0.38431225,
    (stat_t)0.38380810, (stat_t)0.38330526, (stat_t)0.38280374, (stat_t)0.38230353, (stat_t)0.38180462, (stat_t)0.38130702, (stat_t)0.38081071, (stat_t)0.38031569,
    (stat_t)0.37982196, (stat_t)0.37932951, (stat_t)0.37883833, (stat_t)0.37834842, (stat_t)0.37785978, (stat_t)0.37737240, (stat_t)0.37688627, (stat_t)0.37640140,
    (stat_t)0.37591777, (stat_t)0.37543538, (stat_t)0.37495423, (stat_t)0.37447431, (stat_t)0.37399562, (stat_t)0.37351815, (stat_t)0.37304189, (stat_t)0.37256685,
    (stat_t)0.37209302, (stat_t)0.37162040, (stat_t)0.37114897, (stat_t)0.37067873, (stat_t)0.37020969, (stat_t)0.36974183, (stat_t)0.36927515, (stat_t)0.36880965,
    (stat_t)0.36834532, (stat_t)0.36788216, (stat_t)0.36742017, (stat_t)0.36695933, (stat_t)0.36649964, (stat_t)0.36604111, (stat_t)0.36558372, (stat_t)0.36512747,
    (stat_t)0.36467236, (stat_t)0.36421839, (stat_t)0.36376554, (stat_t)0.36331382, (stat_t)0.36286322, (stat_t)0.36241373, (stat_t)0.36196536, (stat_t)0.36151809,
    (stat_t)0.36107193, (stat_t)0.36062687, (stat_t)0.36018291, (stat_t)0.35974003, (stat_t)0.35929825, (stat_t)0.35885754, (stat_t)0.35841792, (stat_t)0.35797937,
    (stat_t)0.35754190, (stat_t)0.35710549, (stat_t)0.35667015, (stat_t)0.35623587, (stat_t)0.35580264, (stat_t)0.35537047, (stat_t)0.35493934, (stat_t)0.35450926,
    (stat_t)0.35408022, (stat_t)0.35365222, (stat_t)0.35322525, (stat_t)0.35279931, (stat_t)0.35237440, (stat_t)0.35195051, (stat_t)0.35152763, (stat_t)0.35110578,
    (stat_t)0.35068493, (stat_t)0.35026509, (stat_t)0.34984626, (stat_t)0.34942843, (stat_t)0.34901159, (stat_t)0.34859574, (stat_t)0.34818089, (stat_t)0.34776702,
    (stat_t)0.34735414, (stat_t)0.34694223, (stat_t)0.34653130, (stat_t)0.34612135, (stat_t)0.34571236, (stat_t)0.34530433, (stat_t)0.34489727, (stat_t)0.34449117,
    (stat_t)0.34408602, (stat_t)0.34368183, (stat_t)0.34327858, (stat_t)0.34287628, (stat_t)0.34247492, (stat_t)0.34207449, (stat_t)0.34167501, (stat_t)0.34127645,
    (stat_t)0.34087883, (stat_t)0.34048213, (stat_t)0.34008635, (stat_t)0.33969149, (stat_t)0.33929755, (stat_t)0.33890452, (stat_t)0.33851240, (stat_t)0.33812118,
    (stat_t)0.33773087, (stat_t)0.33734146, (stat_t)0.33695295, (stat_t)0.33656532, (stat_t)0.33617859, (stat_t)0.33579275, (stat_t)0.33540780, (stat_t)0.33502372,
    (stat_t)0.33464052, (stat_t)0.33425820, (stat_t)0.33387675, (stat_t)0.33349617, (stat_t)0.33311646, (stat_t)0.33273761, (stat_t)0.33235962, (stat_t)0.33198249,
    (stat_t)0.33160622, (stat_t)0.33123079, (stat_t)0.33085622, (stat_t)0.33048249, (stat_t)0.33010961, (stat_t)0.32973756, (stat_t)0.32936636, (stat_t)0.32899598,
    (stat_t)0.32862644, (stat_t)0.32825773, (stat_t)0.32788985, (stat_t)0.32752279, (stat_t)0.32715655, (stat_t)0.32679113, (stat_t)0.32642652, (stat_t)0.32606273,
    (stat_t)0.32569975, (stat_t)0.32533757, (stat_t)0.32497620, (stat_t)0.32461563, (stat_t)0.32425586, (stat_t)0.32389688, (stat_t)0.32353870, (stat_t)0.32318132,
    (stat_t)0.32282472, (stat_t)0.32246890, (stat_t)0.32211387, (stat_t)0.32175962, (stat_t)0.32140615, (stat_t)0.32105346, (stat_t)0.32070153, (stat_t)0.32035038,
    (stat_t)0.32000000, (stat_t)0.31965038, (stat_t)0.31930153, (stat_t)0.31895343, (stat_t)0.31860610, (stat_t)0.31825952, (stat_t)0.31791369, (stat_t)0.31756862,
    (stat_t)0.31722429, (stat_t)0.31688071, (stat_t)0.31653787, (stat_t)0.31619577, (stat_t)0.31585441, (stat_t)0.31551379, (stat_t)0.31517390, (stat_t)0.31483474,
    (stat_t)0.31449631, (stat_t)0.31415861, (stat_t)0.31382164, (stat_t)0.31348538, (stat_t)0.31314985, (stat_t)0.31281503, (stat_t)0.31248093, (stat_t)0.31214754,
    (stat_t)0.31181486, (stat_t)0.31148289, (stat_t)0.31115163, (stat_t)0.31082107, (stat_t)0.31049121, (stat_t)0.31016205, (stat_t)0.30983359, (stat_t)0.30950582,
    (stat_t)0.30917874, (stat_t)0.30885236, (stat_t)0.30852666, (stat_t)0.30820166, (stat_t)0.30787733, (stat_t)0.30755369, (stat_t)0.30723072, (stat_t)0.30690844,
    (stat_t)0.30658683, (stat_t)0.30626589, (stat_t)0.30594562, (stat_t)0.30562603, (stat_t)0.30530710, (stat_t)0.30498883, (stat_t)0.30467123, (stat_t)0.30435429,
    (stat_t)0.30403800, (stat_t)0.30372238, (stat_t)0.30340741, (stat_t)0.30309309, (stat_t)0.30277942, (stat_t)0.30246640, (stat_t)0.30215403, (stat_t)0.30184230,
    (stat_t)0.30153121, (stat_t)0.30122077, (stat_t)0.30091096, (stat_t)0.30060179, (stat_t)0.30029326, (stat_t)0.29998535, (stat_t)0.29967808, (stat_t)0.29937144,
    (stat_t)0.29906542, (stat_t)0.29876003, (stat_t)0.29845526, (stat_t)0.29815111, (stat_t)0.29784759, (stat_t)0.29754468, (stat_t)0.29724238, (stat_t)0.29694070,
    (stat_t)0.29663963, (stat_t)0.29633917, (stat_t)0.29603932, (stat_t)0.29574007, (stat_t)0.29544143, (stat_t)0.29514339, (stat_t)0.29484595, (stat_t)0.29454912,
    (stat_t)0.29425287, (stat_t)0.29395723, (stat_t)0.29366217, (stat_t)0.29336771, (stat_t)0.29307384, (stat_t)0.29278056, (stat_t)0.29248786, (stat_t)0.29219575,
    (stat_t)0.29190422, (stat_t)0.29161327, (stat_t)0.29132290, (stat_t)0.29103311, (stat_t)0.29074390, (stat_t)0.29045525, (stat_t)0.29016719, (stat_t)0.28987969,
    (stat_t)0.28959276, (stat_t)0.28930640, (stat_t)0.28902060, (stat_t)0.28873537, (stat_t)0.28845070, (stat_t)0.28816660, (stat_t)0.28788305, (stat_t)0.28760006,
    (stat_t)0.28731762, (stat_t)0.28703574, (stat_t)0.28675441, (stat_t)0.28647363, (stat_t)0.28619340, (stat_t)0.28591372, (stat_t)0.28563459, (stat_t)0.28535600,
    (stat_t)0.28507795, (stat_t)0.28480045, (stat_t)0.28452348, (stat_t)0.28424705, (stat_t)0.28397116, (stat_t)0.28369580, (stat_t)0.28342098, (stat_t)0.28314669,
    (stat_t)0.28287293, (stat_t)0.28259970, (stat_t)0.28232699, (stat_t)0.28205481, (stat_t)0.28178316, (stat_t)0.28151203, (stat_t)0.28124142, (stat_t)0.28097133,
    (stat_t)0.28070175, (stat_t)0.28043270, (stat_t)0.28016416, (stat_t)0.27989613, (stat_t)0.27962862, (stat_t)0.27936162, (stat_t)0.27909512, (stat_t)0.27882914,
    (stat_t)0.27856366, (stat_t)0.27829868, (stat_t)0.27803421, (stat_t)0.27777024, (stat_t)0.27750678, (stat_t)0.27724381, (stat_t)0.27698134, (stat_t)0.27671936,
    (stat_t)0.27645788, (stat_t)0.27619690, (stat_t)0.27593641, (stat_t)0.27567640, (stat_t)0.27541689, (stat_t)0.27515787, (stat_t)0.27489933, (stat_t)0.27464128,
    (stat_t)0.27438371, (stat_t)0.27412662, (stat_t)0.27387002, (stat_t)0.27361389, (stat_t)0.27335825, (stat_t)0.27310308, (stat_t)0.27284839, (stat_t)0.27259417,
    (stat_t)0.27234043, (stat_t)0.27208715, (stat_t)0.27183435, (stat_t)0.27158202, (stat_t)0.27133015, (stat_t)0.27107876, (stat_t)0.27082782, (stat_t)0.27057736,
    (stat_t)0.27032735, (stat_t)0.27007781, (stat_t)0.26982872, (stat_t)0.26958010, (stat_t)0.26933193, (stat_t)0.26908422, (stat_t)0.26883697, (stat_t)0.26859016,
    (stat_t)0.26834382, (stat_t)0.26809792, (stat_t)0.26785247, (stat_t)0.26760747, (stat_t)0.26736292, (stat_t)0.26711882, (stat_t)0.26687516, (stat_t)0.26663195,
    (stat_t)0.26638918, (stat_t)0.26614685, (stat_t)0.26590496, (stat_t)0.26566351, (stat_t)0.26542250, (stat_t)0.26518192, (stat_t)0.26494179, (stat_t)0.26470208,
    (stat_t)0.26446281, (stat_t)0.26422397, (stat_t)0.26398556, (stat_t)0.26374759, (stat_t)0.26351004, (stat_t)0.26327291, (stat_t)0.26303622, (stat_t)0.26279995,
    (stat_t)0.26256410, (stat_t)0.26232868, (stat_t)0.26209368, (stat_t)0.26185910, (stat_t)0.26162494, (stat_t)0.26139119, (stat_t)0.26115787, (stat_t)0.26092496,
    (stat_t)0.26069246, (stat_t)0.26046038, (stat_t)0.26022872, (stat_t)0.25999746, (stat_t)0.25976662, (stat_t)0.25953618, (stat_t)0.25930615, (stat_t)0.25907653,
    (stat_t)0.25884732, (stat_t)0.25861851, (stat_t)0.25839011, (stat_t)0.25816211, (stat_t)0.25793451, (stat_t)0.25770731, (stat_t)0.25748051, (stat_t)0.25725411,
    (stat_t)0.25702811, (stat_t)0.25680251, (stat_t)0.25657730, (stat_t)0.25635248, (stat_t)0.25612806, (stat_t)0.25590404, (stat_t)0.25568040, (stat_t)0.25545715,
    (stat_t)0.25523430, (stat_t)0.25501183, (stat_t)0.25478975, (stat_t)0.25456805, (stat_t)0.25434675, (stat_t)0.25412582, (stat_t)0.25390528, (stat_t)0.25368512,
    (stat_t)0.25346535, (stat_t)0.25324595, (stat_t)0.25302693, (stat_t)0.25280830, (stat_t)0.25259003, (stat_t)0.25237215, (stat_t)0.25215464, (stat_t)0.25193751,
    (stat_t)0.25172075, (stat_t)0.25150436, (stat_t)0.25128834, (stat_t)0.25107270, (stat_t)0.25085742, (stat_t)0.25064252, (stat_t)0.25042798, (stat_t)0.25021381
    };
    #pragma HLS BIND_STORAGE variable=inv_lut type=rom_1p impl=lutram

    // K4 coefficients: gamma=1.25, delta=0.05
    // 1.25 * x = x + (x >> 2)  → shift-add instead of multiply
    const narrow_t delta = (narrow_t)0.05;

    inv_narrow_t inv_stat;

    for (int i = 0; i < CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LATENCY min=20 max=25
        
        int slot = i & (CHUNKS_PER_BLOCK - 1);
        
        // At start of each block: read stat, do LUT lookup inline
        if (slot == 0) {
            nacc_t x = (nacc_t)stat_stream.read();
            nacc_t x_shifted = x - (nacc_t)0.5;
            // scale = 1024/3.5 ≈ 292.5 = 256 + 32 + 4 + 0.5 → shift-add
            nacc_t sx = x_shifted;
            int idx = (int)((sx << 8) + (sx << 5) + (sx << 2) + (sx >> 1));
            if (idx < 0) idx = 0;
            if (idx > 1023) idx = 1023;
            inv_stat = (inv_narrow_t)inv_lut[idx];
        }
        
        vec32_t in_vec = s1_stream.read();
        vec32_t out_vec;
        #pragma HLS ARRAY_PARTITION variable=out_vec.data complete

        for (int p = 0; p < VEC_SIZE; p++) {
            #pragma HLS UNROLL
            // K3 multiply
            data_narrow_t d = (data_narrow_t)in_vec.data[p];
            ap_fixed<32, 6> product = d * inv_stat;
            #pragma HLS bind_op variable=product op=mul impl=dsp latency=3
            
            // K4 postprocess: 1.25 * x + 0.05 = x + (x >> 2) + 0.05
            nacc_t px = (nacc_t)product;
            nacc_t k4_sum = px + (px >> 2) + (nacc_t)delta;
            narrow_t z = (narrow_t)k4_sum;
            out_vec.data[p] = clamp_n(z, (narrow_t)0, (narrow_t)7.9);
        }
        out_stream.write(out_vec);
    }
}

// ============================================================================
// 5. AXI Write (unchanged)
// ============================================================================

void axi_write(hls::stream<vec32_t>& out_stream, data_t out[N]) {
    #pragma HLS INLINE off
    ultra_wide_t *flat_out = (ultra_wide_t*)out;
    
    for (int k = 0; k < CHUNKS; k++) {
        #pragma HLS PIPELINE II=1
        vec32_t vec = out_stream.read();
        ultra_wide_t raw_chunk = 0;
        
        for (int p = 0; p < VEC_SIZE; p++) {
            #pragma HLS UNROLL
            data_t val = vec.data[p];
            unsigned int raw32 = *(unsigned int*)(&val);
            raw_chunk.range(31 + 32*p, 32*p) = raw32;
        }
        flat_out[k] = raw_chunk;
    }
}

// ============================================================================
// Top-level with Original Signature & DATAFLOW
// ============================================================================

void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in offset=slave bundle=gmem0 register
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1 register
    #pragma HLS interface s_axilite port=return

    // Streams: only 4 needed now (down from 8)
    static hls::stream<vec32_t> raw_K1("raw_K1");
    static hls::stream<vec32_t> raw_K2("raw_K2");
    static hls::stream<vec32_t> s1_K3("s1_K3");
    static hls::stream<nstat_t> stat_K3("stat_K3");
    static hls::stream<vec32_t> out_stream("out_stream");

    // Wide FIFOs in BRAM, narrow FIFOs in SRL
    #pragma HLS BIND_STORAGE variable=raw_K1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=raw_K2 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=s1_K3 type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=out_stream type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stat_K3 type=fifo impl=srl

    #pragma HLS STREAM variable=raw_K1 depth=4
    #pragma HLS STREAM variable=raw_K2 depth=4
    #pragma HLS STREAM variable=s1_K3 depth=64
    #pragma HLS STREAM variable=stat_K3 depth=4
    #pragma HLS STREAM variable=out_stream depth=4

    #pragma HLS DATAFLOW
    
    axi_read_split(in, raw_K1, raw_K2);
    K1_fused(raw_K1, s1_K3);
    K2_fused(raw_K2, stat_K3);
    K3_fused(s1_K3, stat_K3, out_stream);
    axi_write(out_stream, out);
}