## Contains 1 quantization file
__quantcommon.py__  
It is the setting and combination of some quantitative means in Brevitas library

## Contains 8 network files

### Two benchmark networks

- __model_Full.py__  
  Full precision network: L3FNet(FP)
- __model_Quant.py__  
  Quantization network: L3FNet

### Eight ablation implementation networks

#### Four strategies are necessary to prove the ablation network

- __model_Undpp.py__  
  the model with disparity partitioning moved back to CC stage: Net_Undpp
- __model_3D.py__  
  the model using 3D convolutions: Net_3D
- __model_99.py__  
 the model with a 9*9 LF image as input: Net_99
- __model_None.py__  
  represents the network with 9*9 LF images, using 3D convolutions without DPP operations: Net_None

#### Four strategies proved the effectiveness of the ablation network

- __model_Unprune.py__  
  models that have slightly expanded channel and layer numbers for FE, CC, CA stages: Net_Unprune
- __model_8bit.py__  
  an 8-bit quantized network model: Net_8bit
- __model_w8bit.py__  
  only the quantized network whose weights are quantized using 8bit: Net_w8bit
- __model_w2bit.py__  
  only the quantized network whose weights are quantized using 2bit: Net_w2bit
