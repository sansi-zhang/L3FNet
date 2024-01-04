
#define AP_INT_MAX_W 72

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "thresh.h"

// defines for network parameters
#define Channels1 180
 #define InnerProdDim 9

            #define SIMD1 1
 #define PE1 18
 #define numReps 2304

void StreamingDataflowPartition_1_VectorVectorActivation_5(hls::stream<ap_uint<72>> &in0,
                hls::stream<ap_uint<72>> &out
                )
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#include "params.h"
#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3
Vector_Vector_Activate_Batch<Channels1, InnerProdDim, SIMD1, PE1, 1, Slice<ap_uint<4>>, Slice<ap_uint<4>>, Identity>
                (in0, out, weights, threshs, numReps, ap_resource_dsp());
}
