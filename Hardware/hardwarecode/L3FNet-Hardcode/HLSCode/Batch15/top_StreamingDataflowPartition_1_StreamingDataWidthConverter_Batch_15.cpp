
#define AP_INT_MAX_W 72

#include "bnn-library.h"

// includes for network parameters
#include "streamtools.h"

// defines for network parameters
#define InWidth 36 
#define OutWidth 72 
#define NumInWords 36864 
#define numReps 1

void StreamingDataflowPartition_1_StreamingDataWidthConverter_Batch_15(hls::stream<ap_uint<36> > &in0, hls::stream<ap_uint<72> > &out)
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingDataWidthConverter_Batch<InWidth, OutWidth, NumInWords>(in0, out, numReps);
}
