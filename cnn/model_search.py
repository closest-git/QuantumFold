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
    class stem_01(nn.Module):
        def __init__(self,config,C_curr, reduction):
            super(Network.stem_01, self).__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            self.nChanel = C_curr
            self.reduction = reduction
        
        def forward(self, x):
            x = self.stem(x)
            return x
        
        # def weight2gene(self):
        #     return "stem_01_{self.nChanel}_{self.reduction}"
        
        def init_weight(self):
            pass

    def __init__(self,config, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.config = config
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps     #number of nodes in each cell
        #self._multiplier = multiplier      #该设计不是很好
        _concat = list(range(2+self._steps-multiplier, self._steps+2))
        self.topo_darts = TopoStruc(self._steps,_concat)    #always has 4 nodes
        #self._concat = list(range(2+self._steps-multiplier, self._steps+2))
        #self._concat = [2+self._steps-1]   #有问题
        
        if False:
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        self.cells.append( Network.stem_01(config,stem_multiplier*C,False) )
        C_curr = C
        reduction_prev = False
        for i in range(layers):
            if layers>=3 and i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            #cell = StemCell(config,steps, multiplier, C_prev_prev, C_prev,C_curr, reduction, reduction_prev)
            topo = self.topo_darts
            cell = StemCell(config,steps, topo, self.cells,C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            #C_prev_prev, C_prev = C_prev, multiplier*C_curr
        C_prev = self.cells[-1].nChanel
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        #print(self.cells)        
        self._initialize_weights()
        #self._initialize_alphas()
        self.title = f"{self.config.legend()}_{self.topo_darts.legend}"
        print("")

    def new(self):
        model_new = Network(self._C, self._num_classes,self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    
    
    def forward(self, input):
        if hasattr(self,"stem"):    #建议删除stem，应该合并到cells
            s0 = s1 = self.stem(input)
        else:
            s0,s1=None,None

        #self.UpdateWeights()
        attention_func= F.softmax if self.config.attention == "softmax" else entmax15
        all_s=[]
        for i, cell in enumerate(self.cells):
            if s0 is None:
               s0 = s1 = cell(input) 
               all_s.extend([s0,s1])
            else:
                result = cell(all_s)
                if self.config.cell_express=="dense":
                    pass                   
                else:       #DARTS只使用前两个cell
                    #s0, s1 = s1, cell(s0, s1)
                    s0, s1 = s1, result
                all_s.append(result)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits    

    def UpdateWeights(self):        #可以删除了
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

    def _initialize_alphas(self):       #可以删除了
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
    
    def NewATT_weights(self,nOP,isReduce):
        if self.config.op_struc=="se":
            ATT_weight = ATT_se(self.config,nOP,self.topo_darts,isReduce) 
        else:
            ATT_weight = ATT_weights(self.config,nOP,self.topo_darts,isReduce)            
        self._arch_parameters.extend(ATT_weight.get_param())
        self.listWeight.append(ATT_weight)
        return ATT_weight

    def _initialize_weights(self):  
        self.listWeight = []
        #self.listWeight = nn.ModuleList()
        self._arch_parameters=[]        
        isShare = self.config.weight_share
        nOP = len(self.config.PRIMITIVES_pool)
        nNode = sum(1 for i in range(self._steps) for n in range(2+i))
        if isShare:
            w_normal = self.NewATT_weights(nOP,False)
            w_reduce = self.NewATT_weights(nOP,True)
        nReduct,nNormal=0,0
        for i, cell in enumerate(self.cells):   
            if type(cell)==Network.stem_01:
                continue
            if not isShare:
                w_cell = self.NewATT_weights(nOP,cell.reduction)
            if cell.reduction:                
                cell.weight = w_reduce if isShare else w_cell
                nReduct=nReduct+1
            else:
                cell.weight = w_normal if isShare else w_cell
                nNormal=nNormal+1
            cell.init_weight()
        print(f"======"*16) 
        for weight in self.listWeight:
            print(f"====== {weight}")
        print(f"====== _arch_parameters={len(self._arch_parameters)} nReduct={nReduct} nNormal={nNormal}")                          

    # def BeforeEpoch(self):
    #     for ATT_weight in self.listWeight:
    #         ATT_weight.BeforeEpoch()
    # def AfterEpoch(self):
    #     for ATT_weight in self.listWeight:
    #         ATT_weight.AfterEpoch()
    #         if ATT_weight.isReduce:     #仅用于兼容darts
    #             self.alphas_reduce = ATT_weight.alphas_
    #         else:
    #             self.alphas_normal = ATT_weight.alphas_

    def arch_parameters(self):
        nzParam = sum(p.numel() for p in self._arch_parameters)
        return self._arch_parameters
    
    

    def genotype(self):
        assert self.config.weight_share
        if self.config.op_struc == "PCC":
            return self.genotype_PCC()      

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
        if self.config.weight_share:
            if self.config.op_struc == "se":
                if not hasattr(self,"alphas_normal"):
                    return "",False
                alphas_normal,alphas_reduce=self.alphas_normal,self.alphas_reduce
            else:
                assert len(self._arch_parameters)==2
                alphas_normal,alphas_reduce=self._arch_parameters[0],self._arch_parameters[1]
        else:            
            return "",False
                
        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = self.topo_darts._concat   #range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype,True

    def genotype_PCC(self):
        if self.config.weight_share:
            assert len(self._arch_parameters)==4
        alphas_normal,alphas_reduce=self._arch_parameters[0],self._arch_parameters[2]
        betas_normal,betas_reduce=self._arch_parameters[1],self._arch_parameters[3]
        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            PRIMITIVES_pool = self.config.PRIMITIVES_pool
            none_index = PRIMITIVES_pool.index('none')
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :]*W2[j]
                edges = sorted(range(i + 2), key=lambda x: -
                               max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_pool.index('none')))[:2]

                #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
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

        
        n = 3
        start = 2
        weightsr2 = F.softmax(betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
        #gene_n1 = _parse_1(F.softmax(alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        #gene_r1 = _parse_1(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
        #assert gene_normal==gene_n1 and gene_reduce==gene_r1

        concat = self.topo_darts._concat   #range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype,True
