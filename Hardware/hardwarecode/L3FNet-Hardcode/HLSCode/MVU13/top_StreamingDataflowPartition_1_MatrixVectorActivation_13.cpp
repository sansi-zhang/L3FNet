
#define AP_INT_MAX_W 1296

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "mvau.hpp"

// defines for network parameters
#define MW1 180
 #define MH1 9

            #define SIMD1 18
 #define PE1 9
 #define WMEM1 10

            #define TMEM1 0
 #define numReps 2304
#define WP1 8


void StreamingDataflowPartition_1_MatrixVectorActivation_13(
                    hls::stream<ap_uint<72>> &in0,
                    hls::stream<ap_uint<1296>> &weights,
                    hls::stream<ap_uint<216>> &out
                    )
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=weights name=weights_V
#pragma HLS stream depth=8 variable=weights
Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, Slice<ap_uint<4>>, Slice<ap_int<24>>, Identity, ap_int<8> >
                (in0, out, weights, PassThroughActivation<ap_int<24>>(), numReps, ap_resource_dsp());
}
