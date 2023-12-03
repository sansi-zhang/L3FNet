import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList



# # ---------------------------
# # Absolution Experiment
# #
# # Net_None
# # the model using 3D convolutions.


class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, groups, bias):
        super(BuildingBlock, self).__init__()
        
        self.conv = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride = stride,
                    dilation = dilation,
                    padding = padding,
                    groups = groups,
                    bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output) 
        return output

    

## Depthwise separable convolution operations
class DSNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias):
        super(DSNetBlock, self).__init__()
        
        self.deepconv = BuildingBlock(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride = stride,
                            dilation = dilation,
                            padding = padding,
                            groups = in_channels,
                            bias=False)
        self.pointconv = BuildingBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride = 1,
                            dilation = 1,
                            padding = 0,
                            groups = 1,
                            bias=False)
    def forward(self, input):
        output = self.deepconv(input)
        output = self.pointconv(output) 
        return output

class Net_3D(nn.Module):
    def __init__(self, cfg):
        super(Net_3D, self).__init__()
        angRes = cfg.angRes 
        
        Gps = 1
        feaCin = 1
        feaC = 7
        feaCout = 7
        BCin = feaCout
        BCout = 16*9
        ACin = BCout  
        AC = 20*9
        ACout = 1
        
        self.mindisp, self.maxdisp = -4, 4
        self.angRes = angRes
        
             
        layers_feature = []
        layers_feature.append(
            BuildingBlock(in_channels=feaCin, out_channels=feaC, kernel_size=(1,3,3), stride=1, dilation=(1, angRes, angRes), 
                          padding=(0, angRes, angRes), groups=Gps, bias=False))
        for _ in range(5):
            layers_feature.append(
                    BuildingBlock(in_channels=feaC, out_channels=feaC, kernel_size=(1,3,3), stride=1, dilation=(1, angRes, angRes), 
                                  padding=(0, angRes, angRes), groups=Gps, bias=False))
        layers_feature.append(
            BuildingBlock(in_channels=feaC, out_channels=feaCout, kernel_size=(1,3,3), stride=1, dilation=(1, angRes, angRes), 
                          padding=(0, angRes, angRes), groups=Gps, bias=False))
        
        
        self.init_feature = nn.Sequential(*layers_feature)
        
        self.BuildCost = nn.Conv3d(in_channels=BCin, out_channels=BCout, kernel_size=(1, angRes, angRes), 
                                   stride=(1, angRes, angRes), dilation=1, padding=0, groups=Gps, bias=False)
        
        layers_agg = []
        layers_agg.append(
            DSNetBlock(in_channels=ACin, out_channels=AC, kernel_size=3, stride=1, dilation=1, padding=1, 
                           bias=False))
        for _ in range(4):
            layers_agg.append(
                    DSNetBlock(in_channels=AC, out_channels=AC, kernel_size=3, stride=1, dilation=1, padding=1, 
                                 bias=False))
        layers_agg.append(
            nn.Sequential(
                BuildingBlock(in_channels=AC, out_channels=AC, kernel_size=3, stride = 1, dilation = 1, padding = 1, groups = AC, 
                              bias=False),
               nn.Conv3d(in_channels=AC, out_channels=ACout, kernel_size=1, stride=1, bias=False)))
        
        self.aggregation = nn.Sequential(*layers_agg)
        
        self.regression = Regression(self.mindisp, self.maxdisp)
  
    def forward(self, x):
        # print(x.shape)
        x = SAI2MacPI_plus(x, self.angRes)
        # print(x.shape)
        init_feat = self.init_feature(x)
        cost_volume = self.BuildCost(init_feat) 
        # print(cost_volume.shape)
        cost = self.aggregation(cost_volume)
        init_disp = self.regression(cost.squeeze(1))# disp:torch.Size([4, 1, 48, 48]) 
        return init_disp



class Regression(nn.Module):
    def __init__(self, mindisp, maxdisp):
        super(Regression, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def forward(self, cost):
        cost = torch.squeeze(cost, dim=1)
        score = self.softmax(cost)              # B, D, H, W
        temp = torch.zeros(score.shape).to(score.device)            # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)     # B, 1, H, W
        # disp:torch.Size([4, 1, 48, 48])
        return disp



def SAI2MacPI_plus(x, angRes):
    # x:torch.Size([4, 1, 432, 432])
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes     # h=w=48
    mindisp = -4
    maxdisp = 4
    # Compute the MacPI for d=0
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    input = torch.cat(tempU, dim=2)
    
    # MacPI of all d are computed on the basis of d=0
    temp = []
    for d in range(mindisp, maxdisp + 1):
        if d < 0:
            dilat = int(abs(d) * angRes + 1)
            pad = int(0.5 * angRes * (angRes - 1) * abs(d))
        if d == 0:
            dilat = 1
            pad = 0
        if d > 0:
            dilat = int(abs(d) * angRes - 1)
            pad = int(0.5 * angRes * (angRes - 1) * abs(d) - angRes + 1)
        mid = nn.Unfold(kernel_size=angRes, dilation=dilat, padding=pad, stride=angRes)(input)
        out_d = nn.Fold(output_size=(hu,wv), kernel_size=angRes, dilation=1, padding=0, stride=angRes)(mid)
        
        temp.append(out_d)
    out = torch.stack(temp, dim=2)
    return out




