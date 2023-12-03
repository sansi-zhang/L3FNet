import argparse
import numpy as np
import os
from qonnx.core.datatype import DataType
from driver_base import ExampleOverlay

# dictionary describing the I/O of the L3FNet-generated accelerator
io_shape_dict = {
    # DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT24']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 336, 336, 9)],
    "oshape_normal" : [(1, 48, 48, 9)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # compiler.
    "ishape_folded" : [(1, 336, 336, 1, 9)],
    "oshape_folded" : [(1, 48, 48, 1, 9)],
    "ishape_packed" : [(1, 336, 336, 1, 9)],
    "oshape_packed" : [(1, 48, 48, 1, 27)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute L3FNet-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--bitfile', help='name of bitfile', default="L3FNet.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output.npy"])
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir

    # instantiate L3FNet accelerator driver and pass batchsize and bitfile
    accel = ExampleOverlay(
        bitfile_name = bitfile,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir
    )

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file(s)
        ibuf_normal = []
        for ifn in inputfile:
            ibuf_normal.append(np.load(ifn))
        obuf_normal = accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        for o, obuf in enumerate(obuf_normal):
            np.save(outputfile[o], obuf)
    
