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
from genotypes import Genotype
import time
from MixedOp import *
from attention_net import *

#cell具有特定的拓扑结构
class TopoStruc():    
    def __init__(self,nNode,_concat,cells=None):
        super(TopoStruc, self).__init__()
        assert nNode==4         #为了和DARTS对比
        self.nNode = nNode
        self.hGraph = [0]
        end,n = 0,2       
        if cells is not None:   #dense connect:
           n = len(cells)      
        for i in range(self.nNode):
            end = end + n;  n += 1
            self.hGraph.append(end)
        self._concat = _concat
        self.legend = ""
        print("======"*18)
        print(f"======\tTopoStruc_{self.nNode} hGraph={self.hGraph}")
        print(f"======\tconcat={_concat}")
        print("======"*18)

    def I_I(self,id):
        p0,p1 = self.hGraph[id],self.hGraph[id+1]
        #print(range(p0,p1))
        return range(p0,p1)
    
    def nMostEdge(self):
        return self.hGraph[self.nNode]
    

'''
    有趣的干细胞
'''
class StemCell(nn.Module):

    #def __init__(self,config, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    def __init__(self,config,steps, topo, cells, C, reduction, reduction_prev):
        super(StemCell, self).__init__()
        assert(len(cells)>=1)
        
        if topo is None:
            self.topo = TopoStruc(steps,[2,3,4,5],cells)
        else:
            self.topo = topo
        self._concat = self.topo._concat
        self.nChanel = len(self._concat)*C
        self.reduction = reduction
        if len(cells)==1:
            C_prev_prev, C_prev = cells[-1].nChanel,cells[-1].nChanel
        else:
            C_prev_prev, C_prev = cells[-2].nChanel,cells[-1].nChanel

        self.config = config
        self.weight=None      
        self.excitation = None#se_channels(self.nChanel,reduction=2)  

        # if self.config.primitive == "p20":
        #     if reduction_prev:
        #         self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        #     else:
        #         self.preprocess0 = nn.Conv2d(C_prev_prev, C, 1, 1, 0, bias=False)
        #     self.preprocess1 = nn.Conv2d(C_prev, C, 1, 1, 0, bias=False)
        # else:
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        #self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                if self.config.op_struc == "PCC":
                    op = MixedOp_PCC(config,C, stride)
                elif self.config.op_struc == "pair":
                    op = MixedOp_pair(config,C, stride)
                elif self.config.op_struc == "se":
                    op = MixedOp_se(config,C, stride)
                else:
                    op = MixedOp(config,C, stride)

                self._ops.append(op)

    def init_weight(self):
        nOP = len(self._ops)
        if self.config.op_struc == "se":
            assert nOP == len(self.weight.nets)
            for i,op in enumerate(self._ops):
                op.se_op = self.weight.nets[i]
        pass


    #def forward(self, s0, s1, weights=None, weights2=None):
    def forward(self, results):
        assert len(results)>=2
        if self.config.op_struc != "se": 
            [weights,weights2] = self.weight.get_weight("forward")
        else:
            weights,weights2 = None,None
        #[weights,weights2] = self.weight.get_weight()   #se_net一样可以返回weight,非常微妙啊

        if True:
            #s0 = self.preprocess0(s0);            s1 = self.preprocess1(s1)
            s0 = self.preprocess0(results[-2]);    s1 = self.preprocess1(results[-1])
        else:
            pass

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if weights2 is not None:
                s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            elif weights is not None:
                s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset+j](h) for j, h in enumerate(states))

            offset += len(states)
            states.append(s)
        out = torch.cat([states[id] for id in self._concat], dim=1)
        if self.excitation is not None:
            out = self.excitation(out)
        return out
        #return torch.cat(states[-self._multiplier:], dim=1)
    
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
    
    # def weight2gene(self):
    #     gene = self.weight.get_gene()
    #     return gene
    
    def get_alpha(self):
        [weights,weights2] = self.weight.get_weight()
        assert weights is not None
        return weights

'''
    需要和Cell合并
'''





