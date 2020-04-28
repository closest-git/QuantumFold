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
from torch.autograd import Variable
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

'''
    有趣的干细胞
'''
class StemCell(nn.Module):


    #多个cell之间可共用weight!!!
    class OP_weights(object):
        def __init__(self,config, nOP,topo):
            super(StemCell.OP_weights, self).__init__()
            self.config = config
            #self.nNode=nNode
            self.topo = topo
            self.hasBeta = self.config.op_struc == "PCC" or  self.config.op_struc == "pair"
            #k = sum(1 for i in range(self.nNode) for n in range(2+i))            
            k = self.topo.hGraph[-1]
            self.alphas_ = Variable(1e-3*torch.randn((k,nOP)).cuda(), requires_grad=True)
            if self.hasBeta:
                self.betas_ = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)  

        def get_weight(self):
            w_a,w_b = F.softmax(self.alphas_, dim=-1),None
            if self.hasBeta:
                if False:                
                    n = 3
                    start = 2
                    weights2 = F.softmax(self.betas_[0:2], dim=-1)
                    for i in range(self.nNode-1):
                        end = start + n
                        tw2 = F.softmax(self.betas_[start:end], dim=-1)
                        start = end
                        n += 1
                        weights2 = torch.cat([weights2,tw2],dim=0)
                    assert end==len(self.betas_)
                else:
                    I_I = self.topo.I_I
                    weights2 = torch.cat( [F.softmax(self.betas_[I_I(id)], dim=-1) for id in range(self.topo.nNode)] ,dim=0)                
                w_b = weights2                
                            
            return [w_a,w_b]
        
        def get_gene(self):
            [weights,weights2] = self.get_weight()
            weights = weights.detach().cpu().numpy()
            if weights2 is not None:
                weights2 = weights2.detach().cpu().numpy()
            PRIMITIVES_pool = self.config.PRIMITIVES_pool
            gene = []

            none_index = PRIMITIVES_pool.index('none')
            nNode = self.topo.nNode
            for i in range(nNode):
                II = self.topo.I_I(i)
                start = self.topo.hGraph[i]     #类似于单刚和总刚
                nEdge = len(II)
                #W = weights[II].copy()
                if weights2 is not None:
                    #W2 = weights2[II].copy()
                    for j in II:
                        weights[j, :] = weights[j, :]*weights2[j]
                edges,cur_gene = [],[]
                for edge in II:                    
                    W_edge = weights[edge].copy()       #print(W_edge)
                    cur_nz = len(weights[edge])
                    k_sort = sorted(range(cur_nz), key=lambda k:W_edge[k])
                    k_sort.remove(none_index)
                    k_best = k_sort[cur_nz-2]
                    cur_min, cur_max = W_edge[k_sort[0]], W_edge[k_best]
                    edges.append(-cur_max)
                    cur_gene.append((PRIMITIVES_pool[k_best], edge-start))
                edges = sorted(range(nEdge), key=lambda k:edges[k]) #Default is ascending
                gene.extend([cur_gene[edges[0]], cur_gene[edges[1]]])                
            return gene
        
        def get_param(self):
            if self.hasBeta:
                return [self.alphas_,self.betas_]
            else:
                return [self.alphas_]

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
        self.attention = ATT_NET(self.nChanel)  

        if self.config.primitive == "p2":
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
            else:
                self.preprocess0 = nn.Conv2d(C_prev_prev, C, 1, 1, 0, bias=False)
            self.preprocess1 = nn.Conv2d(C_prev, C, 1, 1, 0, bias=False)
        else:
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
                else:
                    op = MixedOp(config,C, stride)
                    #op = BinaryOP(C, stride)

                self._ops.append(op)
    


    #def forward(self, s0, s1, weights=None, weights2=None):
    def forward(self, results):
        assert len(results)>=2
        if self.config.weights == "cys":
            [weights,weights2] = self.weight.get_weight()
        else:
            if weights is None:
                weights = self.weights
            if weights2 is None:
                weights2 = self.weights2
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
            else:
                s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        out = torch.cat([states[id] for id in self._concat], dim=1)
        #out = self.attention(out)
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
    
    def weight2gene(self):
        gene = self.weight.get_gene()
        return gene

'''
    需要和Cell合并
'''
def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class Celler(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Celler, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


