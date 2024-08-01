# FPGA-based Low-bit and Low-power Fast LF Image depth estimation

<img src="./Figure/paper_picture/Net.jpg" alt="L3FNet Network and Custom Data Flow" style="max-width: 60%;">

## Software Preparation

### Requirement

- PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.8, cuda=11.0.
- A GPU with enough memory

### Datasets

- We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de/) for details.

### Path structure

- ./dataset  
  - training  
  Location of the training data.  
  - validation  
  Verify where the data is stored.
- ./Figure  
  - paper_picture  
  Images from the paper.  
  - hardware_picture  
  Hardware design picture.
- ./Hardware  
A file containing a series of hardware for the L3FNe and ablation experimental groups.  
  - L3FNet  
    It contains the bit files and the hwh files for hardware, and the project code for PYNQ implementation.  
  - Net_prune  
    Contains the bit files and the hwh files for hardware.  
  - Net_w2bit  
    Contains the bit files and the hwh files for hardware.  
  - Net_w8bit  
    Contains the bit files and the hwh files for hardware.  
- ./implement  
L3FNet implementation files and data preprocessing file on Pytorch.
- ./jupyter  
Network execution scripts, as well as some algorithm implementation scripts.
- ./model  
Network and regular functions to call.
- ./param  
The checkpoint of the networks is stored here.

- ./Results  
Store network test results, pfm files and converted png files.  
  - our network  
    - Net_Full  
    - Net_Quant  
  - Necessity analysis  
    - Net_3D  
    - Net_99  
    - Net_Undpp  
  - Performance improvement analysis
    - Net_Unprune  
    - Net_8bit  
    - Net_w2bit  
    - Net_w8bit  
    - Net_prune  

### Train

- Set the hyper-parameters in parse_args() if needed. We have provided our default settings in the realeased codes.
- You can train the network by calling implement.py and giving the mode attribute to train.  
    ``` python ../implement/implement.py --net Net_Full  --n_epochs 3000 --mode train --device cuda:1 ```

- Checkpoint will be saved to ./param/'NetName'.
  
### Valition and Test

- After loading the weight file used by your domain, you can call implement.py and giving the mode attribute to valid or test.
- The result files (i.e., scene_name.pfm) will be saved to ./Results/'NetName'.

### Results

#### Contrast with the state-of-the-art work

<img src='./Figure/paper_picture/Top.png'  style="max-width: 40%;">

<img src='./Figure/paper_picture/compare_table.png'  style="max-width: 50%;">

## Hardware Preparation

### Hardware Requirement

- ZCU104 platform
- A memory card with PYNQ installed.  
  For details on the initialization of PYNQ on ZCU104, please refer to the Chinese version of the blog "[PYNQ](https://blog.csdn.net/m0_52279000/article/details/129396434?spm=1001.2014.3001.5501)".
- Vivado Tool Kit (vivado, HLS, etc.)
- An Ubuntu with more than 16GB of memory (the Vivado tool is faster when used in Ubuntu)


### Hardware overall
<img src='./Figure/paper_picture/hardwareoverall.png'  style="max-width: 50%;">

### Hardware Schematic Diagram
See ```'./Figure/hardware_picture/top.pdf' ```

### Hardware Resource Consump
<img src='./Figure/hardware_picture/L3FNet2.png' style="max-width: 50%;">

# Citiation
If you find this work helpful, please consider citing:  
Our paper is currently under submission
<!-- ``` cite
@Article{L3FNet,
    author    = {Jie Li, Chuanlun Zhang, Wenxuan Yang, Heng Li, Xiaoyan Wang, Yiguang Liu},
    title     = {FPGA-based Low-bit and Low-power Fast LF Image depth estimation},
    journal   = {}, 
    year      = {2023},   
}
``` -->

# Contact
Welcome to raise issues or email to Chuanlun Zhang(specialzhangsan@gmail.com or zcl_20000718@163.com) for any question regarding this work

