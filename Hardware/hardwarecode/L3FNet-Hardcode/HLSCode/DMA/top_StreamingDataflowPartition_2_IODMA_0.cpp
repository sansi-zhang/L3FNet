
#define AP_INT_MAX_W 3456

#include "bnn-library.h"

// includes for network parameters
#include "dma.h"
#include "streamtools.h"

// defines for network parameters
#define NumBytes1 62208
#define DataWidth1 128


void StreamingDataflowPartition_2_IODMA_0(hls::stream<ap_uint<216> > &in0, ap_uint<128> *out, unsigned int numReps)
{
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE axis port=in0 name=in0_V
#pragma HLS INTERFACE m_axi offset=slave port=out
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS DATAFLOW
hls::stream<ap_uint<3456> > in2lcm;
hls::stream<ap_uint<128> > lcm2dma;
StreamingDataWidthConverter_Batch<216, 3456, 2304>(in0, in2lcm, numReps);
StreamingDataWidthConverter_Batch<3456, 128, 144>(in2lcm, lcm2dma, numReps);
Stream2Mem_Batch<DataWidth1, NumBytes1>(lcm2dma, out, numReps);
}
