'''
@Author: Yingshi Chen

@Date: 2020-04-18 20:47:51
@
# Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
import time

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

#PARTIAL CHANNEL CONNECTIONS FOR M EMORY -E FFICIENT A RCHITECTURE S EARCH
class MixedOp_PCC(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp_PCC, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in PRIMITIVES:
            op = OPS[primitive](C // 4, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // 4, affine=False))
            self._ops.append(op)
        nOP = len(self._ops)
        self.desc = f"MixedOp_{nOP}_C{C}_stride{stride}"

    def __repr__(self):
        desc = f"{self._ops}"               
        return self.desc

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2//4, :, :]
        xtemp2 = x[:,  dim_2//4:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, 4)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans

