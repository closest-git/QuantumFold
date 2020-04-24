import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from sparse_max import sparsemax, sparsemoid, entmoid15, entmax15
#from genotypes import PRIMITIVES
from genotypes import Genotype
import time
from MixedOp import *
from Cell import *

class Network(nn.Module):

    def __init__(self,config, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.config = config
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps     #number of nodes in each cell
        self._multiplier = multiplier
        #self.attention_func = F.softmax  # 
        #self.attention_func = entmax15      #有意思，差不多
        #self.attention_func = lambda x,dim: x

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(config,steps, multiplier, C_prev_prev, C_prev,C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        #print(self.cells)
        if self.config.weights=="cys":
            self._initialize_weights()
        else:
            self._initialize_alphas()
        share = "" if self.config.weight_share else "$$$"
        attention = self.config.attention[0:3]
        self.title = f"\"{self.config.weights}{share}_{self.config.op_struc}_{self.config.primitive}_{attention}\""
        print("")

    def new(self):
        model_new = Network(self._C, self._num_classes,self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward_V1(self, input):
        # t0=time.time()
        s0 = s1 = self.stem(input)
 
        self.UpdateWeights()
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)            
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        #print(f"Network::forward T={time.time()-t0:.3f}")
        return logits
    
    def forward(self, input):
        if self.config.op_struc != "darts":
            return self.forward_V1(input)

        s0 = s1 = self.stem(input)
        self.UpdateWeights()
        attention_func= F.softmax if self.config.attention == "softmax" else entmax15

        for i, cell in enumerate(self.cells):
            if False:
                if cell.reduction:
                    weights = attention_func(self.alphas_reduce, dim=-1)
                else:
                    weights = attention_func(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def forward_v0(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for i in range(self._steps-1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2,tw2],dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for i in range(self._steps-1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2,tw2],dim=0)
            s0, s1 = s1, cell(s0, s1, weights,weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def UpdateWeights(self):
        if self.config.weights == "cys":
            return

        attention_func = F.softmax
        contiguous = False     #./dump/conti_1.info,./dump/conti_2.info
        if contiguous:
            a_reduce = self.alphas_reduce.contiguous().view(-1)
            a_normal = self.alphas_normal.contiguous().view(-1)
            a_reduce = entmax15(a_reduce, dim=-1)
            a_normal = entmax15(a_normal, dim=-1)
            a_reduce = a_reduce.view_as(self.alphas_reduce)
            a_normal = a_normal.view_as(self.alphas_normal)
        else:
            a_reduce = attention_func(self.alphas_reduce, dim=-1)
            a_normal = attention_func(self.alphas_normal, dim=-1)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                cell.weights = a_reduce 
                beta = self.betas_reduce
            else:
                cell.weights = a_normal 
                beta = self.betas_normal
            
            n = 3
            start = 2
            weights2 = F.softmax(beta[0:2], dim=-1)
            for i in range(self._steps-1):
                end = start + n
                tw2 = F.softmax(beta[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2,tw2],dim=0)
            cell.weights2 = weights2

        #a_reduce = self.attention_func(self.alphas_reduce, dim=-1)
        #a_normal = self.attention_func(self.alphas_normal, dim=-1)
        return 

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(2+i for i in range(self._steps) )
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        nCell = len(self.cells)
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        if self.config.op_struc != "darts":
            self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
            self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
            self._arch_parameters.append(self.betas_normal)
            self._arch_parameters.append(self.betas_reduce)  

    def _initialize_weights(self):
        self._arch_parameters=[]
        isShare = self.config.weight_share
        nOP = len(self.config.PRIMITIVES_pool)
        nNode = sum(1 for i in range(self._steps) for n in range(2+i))
        if isShare:
            w_normal = Cell.OP_weights(self.config,nOP,self._steps)
            w_reduce = Cell.OP_weights(self.config,nOP,self._steps)
            self._arch_parameters.extend(w_normal.get_param())
            self._arch_parameters.extend(w_reduce.get_param())
        nReduct,nNormal=0,0
        for i, cell in enumerate(self.cells):   
            if not isShare:
                w_cell = Cell.OP_weights(self.config,nOP,self._steps)
                self._arch_parameters.extend(w_cell.get_param())
            if cell.reduction:                
                cell.weight = w_reduce if isShare else w_cell
                nReduct=nReduct+1
            else:
                cell.weight = w_normal if isShare else w_cell
                nNormal=nNormal+1
        print(f"====== _arch_parameters={len(self._arch_parameters)} nReduct={nReduct} nNormal={nNormal}")                          

    def arch_parameters(self):
        nzParam = sum(p.numel() for p in self._arch_parameters)
        return self._arch_parameters
    
    

    def genotype(self):
        def _parse(weights):
            PRIMITIVES_pool = self.config.PRIMITIVES_pool
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_pool.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES_pool.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES_pool[k_best], j))
                start = end
                n += 1
            return gene
        #alphas_normal,alphas_reduce=self.alphas_normal,self.alphas_reduce
        alphas_normal,alphas_reduce=self._arch_parameters[0],self._arch_parameters[1]
        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def genotype_v1(self):

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            none_index = PRIMITIVES.index('none')
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :]*W2[j]
                edges = sorted(range(i + 2), key=lambda x: -
                               max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]

                #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        
        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
        gene_n1 = _parse_1(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_r1 = _parse_1(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
        assert gene_normal==gene_n1 and gene_reduce==gene_r1

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
