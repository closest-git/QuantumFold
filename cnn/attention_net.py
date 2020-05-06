'''
@Author: Yingshi Chen

@Date: 2020-04-27 18:30:01
@
# Description: 
'''
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from sparse_max import sparsemax, sparsemoid, entmoid15, entmax15
from genotypes import Genotype
import time
from MixedOp import *
from torch.autograd import Variable
import seaborn as sns;      sns.set()
import matplotlib.pyplot as plt

class se_channels(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_channels, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    #einsum is more elegant than code at https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def forward_verify(self, x,out_0):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        out = torch.einsum('bcxy,bc->bcxy', x,y)        
        dist = torch.dist(out,out_0,2)
        assert dist==0
        return 

    #elegant code from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        self.forward_verify(x,out)        
        return out

class se_operate(nn.Module):
    def __init__(self, nOP, reduction=2):
        super(se_operate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #The number of output features is equal to the number of input planes.
        self.nOP,reduction = nOP,2
        self.fc = nn.Sequential(
            nn.Linear(nOP, nOP // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nOP // reduction, nOP, bias=False),
            #nn.Sigmoid()
            nn.Softmax()
        )  
        self.desc=f"se_operate_{reduction}"
        self.InitAlpha()
    
    def __repr__(self):
        return self.desc

    def InitAlpha(self):
        self.nStep = 0
        self.alpha = torch.zeros(self.nOP)
    
    def UpdateAlpha(self):
        self.alpha=self.alpha/self.nStep
        a = torch.sum(self.alpha).item()
        assert np.isclose(a, 1) 

    #elegant code from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def forward(self, listOPX):
        assert len(listOPX)==self.nOP
        y_list=[]
        for i,opx in enumerate(listOPX):
            y = torch.mean(self.avg_pool(opx).squeeze(),dim=1)
            y_list.append(y)
        y = torch.stack( y_list ,dim=1) 
        w = self.fc(y)
        m_ = torch.mean(w,dim=0) 
        #assert np.isclose(torch.sum(m_).item(), 1) 
        if False:       #原则上应停止累加
            pass
        else:
            self.alpha = self.alpha+ m_.cpu()
            self.nStep = self.nStep+1           
        out = 0
        for i,opx in enumerate(listOPX):
            w_i = w[:,i:i+1].squeeze()
            out = out+torch.einsum('bcxy,b->bcxy',opx,w_i)          
        
        return out

#多个cell之间可共用weight!!!
class ATT_weights(object):
    def __init__(self,config, nOP,topo,isReduce=False):
        super(ATT_weights, self).__init__()
        self.config = config
        self.nets = None
        self.topo = topo
        self.isReduce = isReduce
        self.hasBeta = self.config.op_struc == "PCC" or  self.config.op_struc == "pair"
        #k = sum(1 for i in range(self.nNode) for n in range(2+i))            
        k = self.topo.hGraph[-1]
        self.desc = f"W[{k},{nOP}]"
        if self.config.op_struc=="se":            
            pass
        else:
            self.alphas_ = Variable(1e-3*torch.randn((k,nOP)).cuda(), requires_grad=True)
            if self.hasBeta:
                self.betas_ = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)          

        if self.hasBeta:   self.desc+="\tbeta"
        if self.isReduce:   self.desc+="\treduce"

    def __repr__(self):
        return self.desc

    # def BeforeEpoch(self):
    #     pass

    # def AfterEpoch(self):
    #     pass

    def step(self):
        pass

    def get_weight(self,purpose="get_gene"):
        if not hasattr(self,"alphas_"):
            # assert False
            return [None,None]

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
    
    def get_gene(self,plot_path=""):
        [weights,weights2] = self.get_weight()
        if weights is None:
            return ""
        weights = weights.detach().cpu().numpy()
        if weights2 is not None:
            weights2 = weights2.detach().cpu().numpy()
        
        nNode = self.topo.nNode
        nEdges = self.topo.nMostEdge()
        PRIMITIVES_pool = self.config.PRIMITIVES_pool
        if plot_path is not None:
            sns.set(font_scale=1)
            fig, ax = plt.subplots(figsize=(8,3))
            g = sns.heatmap(weights.T,square=True, cmap='coolwarm', ax=ax)       #, annot=True
            g.set_yticklabels(PRIMITIVES_pool, rotation=0)
            g.set_xticklabels([i+1 for i in range(nEdges)],rotation=0)  #rotation=45
            fig.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            #plt.show()
            plt.close("all")
        gene = []

        none_index = PRIMITIVES_pool.index('none')
        
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
        param_list = []
        if self.nets is not None:
            for net in self.nets:
                for name, param in net.named_parameters():
                    param_list.append(param)         
            return param_list   
        if self.hasBeta:
            return [self.alphas_,self.betas_]
        else:
            return [self.alphas_]

class ATT_se(ATT_weights):
    def __init__(self,config, nOP,topo,isReduce=False):
        super(ATT_se, self).__init__(config, nOP,topo,isReduce)
        self.nets = [se_operate(nOP) for i in range(self.topo.nMostEdge())]
        self.nets = nn.ModuleList(self.nets)
        self.desc+=f"\t\"{self.nets[0]}\"x{len(self.nets)}"

    def __repr__(self):
        return self.desc

    # def BeforeEpoch(self):
    #     for net in self.nets:
    #         net.BeforeEpoch()

    def step(self):
        list_alpha=[]            
        for net in self.nets:
            net.UpdateAlpha()
            list_alpha.append(net.alpha)
        self.alphas_ = torch.stack(list_alpha,dim=0)
            #print("")
        for net in self.nets:   #重置
            net.InitAlpha()


    def get_weight(self):
        if not hasattr(self,"alphas_"):
            # assert False
            return [None,None]

        return [self.alphas_,None]
        
        