
#define AP_INT_MAX_W 252

#include "bnn-library.h"

// includes for network parameters
#include "streamtools.h"

// defines for network parameters
#define InWidth 84 
#define OutWidth 252 
#define NumInWords 338688 
#define numReps 1

void StreamingDataflowPartition_1_StreamingDataWidthConverter_Batch_6(hls::stream<ap_uint<84> > &in0, hls::stream<ap_uint<252> > &out)
{
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE axis port=out name=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingDataWidthConverter_Batch<InWidth, OutWidth, NumInWords>(in0, out, numReps);
}
