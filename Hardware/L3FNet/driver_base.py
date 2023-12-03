import numpy as np
import os
import time
from pynq import Overlay, allocate
from pynq.ps import Clocks
from qonnx.core.datatype import DataType

from datatrans.util.data_packing import (
    py_to_packed_bytearray,
    packed_bytearray_to_py,
)

# # --------------
# # Driver base class for generated dataflow accelerators.
# #


class ExampleOverlay(Overlay):
    def __init__(
        self,
        bitfile_name,
        io_shape_dict,
        batch_size=1,
        fclk_mhz=100.0,
        device=None,
        download=True,
        runtime_weight_dir="runtime_weights/",
    ):
        """Initialize the L3FNet accelerator.

        Parameters
        ----------
        bitfile_name: str
            Path to accelerator .bit/.xclbin file
        io_shape_dict: dict
            Dictionary with particulars of the generated accelerator
        batch_size: int
            Maximum batch size in driver (hardware batchsize is always 1)
        fclk_mhz: float
            Override the clock frequency, only possible for Zynq.
        device: pynq.Device
            Which PYNQ device to use, None for default.
        download: bool
            Whether to flash the bitstream.
        runtime_weight_dir: str
            Path to runtime weights folder.
        """
        super().__init__(bitfile_name, download=download, device=device)
        self.runtime_weight_dir = runtime_weight_dir
        self._io_shape_dict = io_shape_dict
        self.ibuf_packed_device = None
        self.obuf_packed_device = None
        self.batch_size = batch_size
        self.fclk_mhz = fclk_mhz
        self.idma = []
        self.odma = []
        self.odma_handle = []
        if "input_dma_name" in io_shape_dict.keys():
            for idma_name in io_shape_dict["input_dma_name"]:
                self.idma.append(getattr(self, idma_name))
        else:
            self.idma = [self.idma0]
        if "output_dma_name" in io_shape_dict.keys():
            for odma_name in io_shape_dict["output_dma_name"]:
                self.odma.append(getattr(self, odma_name))
        else:
            self.odma = [self.odma0]

        # set the clock frequency as specified by user during transformations
        if self.fclk_mhz > 0:
            Clocks.fclk0_mhz = self.fclk_mhz
        # load any external + runtime weights
        self.load_external_weights()
        self.load_runtime_weights()

    def load_external_weights(self):
        """Load any existing external (DRAM) weights from the specified dir into the
        appropriate layer of the accelerator. Note that this must be enabled
        during the accelerator build process. The weights directory
        is specified as the class member ``runtime_weight_dir``. External (DRAM)
        weights are one .npy file per layer.
        """

        self.external_weights = []
        w_filenames = []
        if not os.path.isdir(self.runtime_weight_dir):
            return
        for (dirpath, dirnames, filenames) in os.walk(self.runtime_weight_dir):
            w_filenames.extend(filenames)

        tmp_weight_dict = {}

        for w_filename in w_filenames:
            if w_filename.endswith(".npy"):
                weight_tensor = np.load(self.runtime_weight_dir + "/" + w_filename)
            else:
                continue

            idma_name = w_filename.split(".")[0]
            tmp_weight_dict[idma_name] = weight_tensor

        for idma_name in tmp_weight_dict.keys():
            if idma_name in self.ip_dict.keys():
                iwdma = getattr(self, idma_name)
                weight_tensor = tmp_weight_dict[idma_name]
                weight_buf = allocate(weight_tensor.shape, dtype=np.uint8)
                weight_buf[:] = weight_tensor
                # weight_buf.sync_to_device()
                weight_buf.flush()

                self.external_weights += [(iwdma, weight_buf, idma_name)]

        if "number_of_external_weights" in self._io_shape_dict:
            hw_ext_weights = self._io_shape_dict["number_of_external_weights"]
            assert len(self.external_weights) == hw_ext_weights, (
                "Number of hardware external weights and number of external "
                + "weight tensors available do not match. \n"
                + "Is runtime_weight_dir pointing to the correct folder?"
            )

    def load_runtime_weights(self, flush_accel=True, verify=True):
        """Load any existing runtime-writable weights from the specified dir into the
        appropriate layer of the accelerator. Note that this must be enabled
        during the accelerator build process. The runtime weights directory
        is specified as the class member ``runtime_weight_dir``. Runtime-writable
        weights are provided as one .dat file per layer.

        Parameters
        ----------
        flush_accel: bool
            Run the accelerator with dummy input after weights are written to
            flush any stale weight data in the weight streamer FIFOs.
        verify: bool
            Whether the written weights will be re-read and verified.
        """
        w_filenames = []
        if not os.path.isdir(self.runtime_weight_dir):
            return
        for (dirpath, dirnames, filenames) in os.walk(self.runtime_weight_dir):
            w_filenames.extend(filenames)
        rt_weight_dict = {}
        for w_filename in w_filenames:
            if w_filename.endswith(".dat"):
                with open(self.runtime_weight_dir + "/" + w_filename, "r") as f:
                    dat = f.read()
            else:
                continue
            layer_w = np.fromiter(
                [int(x, 16) for x in dat.strip().split()], dtype=np.uint32
            )
            sdp_ind = int(w_filename.split("_")[0])
            layer_ind = int(w_filename.split("_")[1])
            rt_weight_dict[(sdp_ind, layer_ind)] = layer_w
        for sdp_ind, layer_ind in rt_weight_dict.keys():
            cand_if_name = "StreamingDataflowPartition_%d/s_axilite_%d" % (
                sdp_ind,
                layer_ind,
            )
            if cand_if_name in self.ip_dict.keys():
                layer_mmio = getattr(
                    getattr(self, "StreamingDataflowPartition_%d" % sdp_ind),
                    "s_axilite_%d" % layer_ind,
                ).mmio
                layer_w = rt_weight_dict[(sdp_ind, layer_ind)]
                layer_mmio.write_mm(0, layer_w.tobytes())
                if verify:
                    new_w = np.copy(layer_mmio.array[: layer_w.shape[0]])
                    assert (layer_w == new_w).all()
        if flush_accel:
            # run accelerator to flush any stale weights from weight streamer FIFOs
            self.execute_on_buffers()

    def idt(self, ind=0):
        return self._io_shape_dict["idt"][ind]

    def odt(self, ind=0):
        return self._io_shape_dict["odt"][ind]

    def ishape_normal(self, ind=0):
        ret = list(self._io_shape_dict["ishape_normal"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_normal(self, ind=0):
        ret = list(self._io_shape_dict["oshape_normal"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def ishape_folded(self, ind=0):
        ret = list(self._io_shape_dict["ishape_folded"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_folded(self, ind=0):
        ret = list(self._io_shape_dict["oshape_folded"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def ishape_packed(self, ind=0):
        ret = list(self._io_shape_dict["ishape_packed"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_packed(self, ind=0):
        ret = list(self._io_shape_dict["oshape_packed"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def num_inputs(self):
        return self._io_shape_dict["num_inputs"]

    @property
    def num_outputs(self):
        return self._io_shape_dict["num_outputs"]

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        # free the old buffers by setting to None
        # (reference counting should care of it)
        if self.ibuf_packed_device is not None:
            self.ibuf_packed_device = None
        if self.obuf_packed_device is not None:
            self.obuf_packed_device = None
        cacheable = True
        self.ibuf_packed_device = []
        self.obuf_packed_device = []
        self.obuf_packed = []
        for i in range(self.num_inputs):
            new_packed_ibuf = allocate(
                shape=self.ishape_packed(i), dtype=np.uint8, cacheable=cacheable
            )
            self.ibuf_packed_device.append(new_packed_ibuf)
        for o in range(self.num_outputs):
            new_packed_obuf = allocate(
                shape=self.oshape_packed(o), dtype=np.uint8, cacheable=cacheable
            )
            self.obuf_packed_device.append(new_packed_obuf)
            self.obuf_packed.append(np.empty_like(new_packed_obuf))

    def fold_input(self, ibuf_normal, ind=0):
        """Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input."""
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal(ind)
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded(ind))
        return ibuf_folded

    def pack_input(self, ibuf_folded, ind=0):
        """Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data."""
        ibuf_packed = py_to_packed_bytearray(
            ibuf_folded,
            self.idt(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed, ind=0):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_py(
            obuf_packed,
            self.odt(ind),
            self.oshape_folded(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )
        return obuf_folded

    def unfold_output(self, obuf_folded, ind=0):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal(ind))
        return obuf_normal

    def copy_input_data_to_device(self, data, ind=0):
        """Copies given input data to PYNQ buffer."""
        np.copyto(self.ibuf_packed_device[ind], data)
        self.ibuf_packed_device[ind].flush()

    def copy_output_data_from_device(self, data, ind=0):
        """Copies PYNQ output buffer from device."""
        self.obuf_packed_device[ind].invalidate()
        np.copyto(data, self.obuf_packed_device[ind])

    def execute_on_buffers(self, asynch=False, batch_size=None):
        """Executes accelerator by setting up the DMA(s) on pre-allocated buffers.
        Blocking behavior depends on the asynch parameter:
        * ``asynch=True`` will block until all transfers are complete.
        * ``asynch=False`` won't block, use ``wait_until_finished()`` to check
           completion

        The optional batch_size parameter can be used to execute on a smaller
        batch than the initialized ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size, "Specified batch_size is too large."
        for o in range(self.num_outputs):
            assert (
                self.odma[o].read(0x00) & 0x4 != 0
            ), "Output DMA %d is not idle" % (o)
        # manually launch IODMAs since signatures are missing
        for iwdma, iwbuf, iwdma_name in self.external_weights:
            iwdma.write(0x10, iwbuf.device_address)
            iwdma.write(0x1C, batch_size)
            iwdma.write(0x00, 1)
        for o in range(self.num_outputs):
            self.odma[o].write(0x10, self.obuf_packed_device[o].device_address)
            self.odma[o].write(0x1C, batch_size)
            self.odma[o].write(0x00, 1)
        for i in range(self.num_inputs):
            self.idma[i].write(0x10, self.ibuf_packed_device[i].device_address)
            self.idma[i].write(0x1C, batch_size)
            self.idma[i].write(0x00, 1)
    
        # blocking behavior depends on asynch parameter
        if asynch is False:
            self.wait_until_finished()

    def wait_until_finished(self):
        "Block until all output DMAs have finished writing."
        # check if output IODMA is finished via register reads
        for o in range(self.num_outputs):
            status = self.odma[o].read(0x00)
            while status & 0x2 == 0:
                status = self.odma[o].read(0x00)
        

    def execute(self, input_npy):
        """Given a single or a list of input numpy array, first perform necessary
        packing and copying to device buffers, execute on accelerator, then unpack
        output and return output numpy array from accelerator."""
        # if single input, convert to list to normalize how we process the input
        if not type(input_npy) is list:
            input_npy = [input_npy]
        assert self.num_inputs == len(
            input_npy
        ), "Not all accelerator inputs are specified."
        for i in range(self.num_inputs):
            ibuf_folded = self.fold_input(input_npy[i], ind=i)
            ibuf_packed = self.pack_input(ibuf_folded, ind=i)
            self.copy_input_data_to_device(ibuf_packed, ind=i)
        self.execute_on_buffers()
        outputs = []
        for o in range(self.num_outputs):
            self.copy_output_data_from_device(self.obuf_packed[o], ind=o)
            obuf_folded = self.unpack_output(self.obuf_packed[o], ind=o)
            obuf_normal = self.unfold_output(obuf_folded, ind=o)
            outputs.append(obuf_normal)
        if self.num_outputs == 1:
            return outputs[0]
        else:
            return outputs


