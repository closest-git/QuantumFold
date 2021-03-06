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
from genotypes import Genotype
import time
import random
import numpy as np
'''
    We propose an alternative and more efficient implementation for partial channel connections. For
    edge (i,j), we do not perform channel sampling at each time of computing o(x i ), but instead choose
    the first 1/K channels of x i for operation mixture directly. To compensate, after x j is obtained, we
    shuffle its channels before using it for further computations. This is the same implementation used
    in ShuffleNet (Zhang et al., 2018), which is more GPU-friendly and thus runs faster.
'''
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self,config, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.config = config

        for primitive in self.config.PRIMITIVES_pool:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive and config.primitive == "c0":
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
        nOP = len(self._ops)
        self.desc = f"MixedOp_{nOP}_C{C}_stride{stride}"

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    def __repr__(self):
        return self.desc

class MixedOp_se(MixedOp):
    def __init__(self,config, C, stride):
        super(MixedOp_se, self).__init__(config, C, stride)  
        nOP = len(self._ops)   
        self.desc = f"SE_op_{nOP}_C{C}_stride{stride}"

    def forward(self, x, alpha=None,beta=None):        
        assert hasattr(self,'se_op')
        se_net = self.se_op
        if self.config.attention=="E2W" and not self.config.search_alpha:
            alpha = self.se_op.alpha
            return super().forward(x,alpha)

        nOP = len(self._ops)
        b, c, _, _ = x.size()

        opx_list = []
        for op in self._ops:
            opx = op(x)            
            opx_list.append(opx)
        out = se_net(opx_list) 
        return out

class MixedOp_pair(nn.Module):
    def __init__(self,config, C, stride):
        super(MixedOp_pair, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.nCand = C//4      #不能取2，极不稳定
        self.config = config
        
        self.isShuffle = True    
        
        for primitive in PRIMITIVES_pool:
            op = OPS[primitive](self.nCand, stride, False)
            #if 'pool' in primitive:            op = nn.Sequential(op, nn.BatchNorm2d(self.nCand, affine=False))
            self._ops.append(op)
        nOP = len(self._ops)
        self.desc = f"MixedOp{self.nCand}_{nOP}_C{C}_stride{stride}"

    def __repr__(self):
        desc = f"{self._ops}"
        return self.desc

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        if False:#random permute提速很少，而准确率远不如shuffle net
            perm = np.arange(dim_2)
            np.random.shuffle(perm)
            x = x[:,perm]
            pass
        
        xtemp = x[:, :  self.nCand, :, :]
        xtemp2 = x[:,  self.nCand:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))        
        # temp1 = 0
        # for w, op in zip(weights, self._ops):
        #     print(f"w={w.item()} op={op} xtemp={xtemp.shape}")
        #     temp1 = temp1 + w * op(xtemp)
        
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        if self.isShuffle:  #晕，关闭之后竟然略有提升
            ans = channel_shuffle(ans, 4)

        return ans

# PARTIAL CHANNEL CONNECTIONS FOR M EMORY -E FFICIENT A RCHITECTURE S EARCH
class MixedOp_PCC(nn.Module):

    def __init__(self,config, C, stride):
        super(MixedOp_PCC, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.config = config

        for primitive in self.config.PRIMITIVES_pool: #PRIMITIVES
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
