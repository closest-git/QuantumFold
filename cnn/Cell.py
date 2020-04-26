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

'''
    有趣的干细胞
'''
class StemCell(nn.Module):
    #多个cell之间可共用weight!!!
    class OP_weights(object):
        def __init__(self,config, nOP,nNode):
            self.config = config
            self.nNode=nNode
            self.hasBeta = self.config.op_struc == "PCC" or  self.config.op_struc == "pair"
            k = sum(1 for i in range(self.nNode) for n in range(2+i))
            self.alphas_ = Variable(1e-3*torch.randn((k,nOP)).cuda(), requires_grad=True)
            if self.hasBeta:
                self.betas_ = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
            

        def get_weight(self):
            w_a,w_b = F.softmax(self.alphas_, dim=-1),None
            if self.hasBeta:
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
                w_b = weights2                
                            
            return [w_a,w_b]
        
        def get_param(self):
            if self.hasBeta:
                return [self.alphas_,self.betas_]
            else:
                return [self.alphas_]

    #def __init__(self,config, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    def __init__(self,config,steps, multiplier, cells, C, reduction, reduction_prev):
        super(StemCell, self).__init__()
        assert(len(cells)>=1)
        self.nChanel = multiplier*C
        self.reduction = reduction
        if len(cells)==1:
            C_prev_prev, C_prev = cells[-1].nChanel,cells[-1].nChanel
        else:
            C_prev_prev, C_prev = cells[-2].nChanel,cells[-1].nChanel

        self.config = config
        self.reduction = reduction
        self.weight=None        

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
        self._multiplier = multiplier

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

        
    def forward(self, s0, s1, weights=None, weights2=None):
        if self.config.weights == "cys":
            [weights,weights2] = self.weight.get_weight()
        else:
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
                if weights2 is not None:
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
    
    def weight2gene(self):
        [weights,weights2] = self.weight.get_weight()
        weights = weights.detach().cpu().numpy()
        if weights2 is not None:
            weights2 = weights2.detach().cpu().numpy()
        PRIMITIVES_pool = self.config.PRIMITIVES_pool
        gene = []
        n = 2   #2,3,4,5... number of edges
        start = 0
        none_index = PRIMITIVES_pool.index('none')
        for i in range(self._steps):
            end = start + n
            W = weights[start:end].copy()
            if weights2 is not None:
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :]*W2[j]
            edges,cur_gene = [],[]
            for edge in range(n):
                #print(W[edge])
                cur_nz = len(W[edge])
                k_sort = sorted(range(cur_nz), key=lambda k:W[edge][k])
                k_sort.remove(none_index)
                k_best = k_sort[cur_nz-2]
                cur_min, cur_max = W[edge][k_sort[0]], W[edge][k_best]
                edges.append(-cur_max)
                cur_gene.append((PRIMITIVES_pool[k_best], edge))
            edges = sorted(range(n), key=lambda k:edges[k]) #Default is ascending
            gene.extend([cur_gene[edges[0]], cur_gene[edges[1]]])
            start = end
            n += 1
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


