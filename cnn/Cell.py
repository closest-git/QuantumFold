'''
@Author: Yingshi Chen

@Date: 2020-04-20 12:52:23
@
# Description: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from sparse_max import sparsemax, sparsemoid, entmoid15, entmax15
from genotypes import PRIMITIVES
from genotypes import Genotype
import time
from MixedOp import *

class Cell(nn.Module):
    def __init__(self,config, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.config = config
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                if self.config.op_struc == "PCC":
                    op = MixedOp_PCC(C, stride)
                else:
                    op = MixedOp(C, stride)
                    #op = BinaryOP(C, stride)
                self._ops.append(op)

        self.weights = None
        self.weights2 = None
        # print(f"{self}")
        
    def forward(self, s0, s1, weights=None, weights2=None):
        if weights is None:
            weights = self.weights
        if weights2 is None:
            weights2 = self.weights2

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if False:   #确实不行 
                no = random.sample(range(len(states)), 1)[0]
                s = self._ops[offset+no](states[no], weights[offset+no]) 
            else:
                if self.config.op_struc == "PCC":
                    s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
                else:
                    s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
    
    def UpdateAttention(self,alpha,beta):
        if self.reduction:
            weights = self.attention_func(self.alphas_reduce, dim=-1)
            n = 3
            start = 2
            weights2 = self.attention_func(self.betas_reduce[0:2], dim=-1)
            for i in range(self._steps-1):
                end = start + n
                tw2 = self.attention_func(self.betas_reduce[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)
        else:
            weights = self.attention_func(self.alphas_normal, dim=-1)
            n = 3
            start = 2
            weights2 = self.attention_func(self.betas_normal[0:2], dim=-1)
            for i in range(self._steps-1):
                end = start + n
                tw2 = self.attention_func(self.betas_normal[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)


