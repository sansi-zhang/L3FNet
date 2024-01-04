
#define AP_INT_MAX_W 3024

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "mvau.hpp"

// defines for network parameters
#define MW1 3087
 #define MH1 144

            #define SIMD1 21
 #define PE1 36
 #define WMEM1 588

            #define TMEM1 0
 #define numReps 2304
#define WP1 4


void StreamingDataflowPartition_1_MatrixVectorActivation_7(
                    hls::stream<ap_uint<84>> &in0,
                    hls::stream<ap_uint<3024>> &weights,
                    hls::stream<ap_uint<576>> &out
                    )
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=weights name=weights_V
#pragma HLS stream depth=8 variable=weights
Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, Slice<ap_uint<4>>, Slice<ap_int<16>>, Identity, ap_int<4> >
                (in0, out, weights, PassThroughActivation<ap_int<16>>(), numReps, ap_resource_lut());
}
