
#define AP_INT_MAX_W 648

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "mvau.hpp"
#include "thresh.h"

// defines for network parameters
#define MW1 180
 #define MH1 180

            #define SIMD1 18
 #define PE1 9
 #define WMEM1 200

            #define TMEM1 20
 #define numReps 2304
#define WP1 4


void StreamingDataflowPartition_1_MatrixVectorActivation_10(
                    hls::stream<ap_uint<72>> &in0,
                    hls::stream<ap_uint<648>> &weights,
                    hls::stream<ap_uint<36>> &out
                    )
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=weights name=weights_V
#pragma HLS stream depth=8 variable=weights
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3
Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, Slice<ap_uint<4>>, Slice<ap_uint<4>>, Identity, ap_int<4> >
                (in0, out, weights, threshs, numReps, ap_resource_dsp());
}
