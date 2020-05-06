'''
@Author: Yingshi Chen

@Date: 2020-02-14 11:59:10
@
# Description: 
'''
class QuantumFold_config:
    def __init__(self,data,random_seed=42):
    
        self.model = "QuantumFold_cnn"
        #self.tree_type = tree_type
        # self.data = data
        self.data_set = "cifar10"      
        # self.lr_base = lr_base
        # self.nLayer = nLayer
        self.seed = random_seed
        #seed_everything(self.seed)
        #self.init_value = init_value  # "random"  "zero"
        #self.choice_func = choice_func
        self.rDrop = 0
        self.custom_legend = None
        #self.feat_info = feat_info
        self.no_attention = False
        self.max_features = None
        self.input_dropout = 0        #YAHOO 0.59253-0.59136 没啥用
        self.num_layers = 1
        self.flatten_output = True
        self.max_out = True
        self.plot_root = "./results/"
        self.plot_train = False
        self.plot_attention = True

        self.op_struc = "darts"           #"PCC" "darts" "pair" "se"
        #self.weights = "cys"              #"cys"  
        self.primitive = "p1"              #"p0"    "p1"  "p2"   "c0"
        self.attention = "softmax"         #"entmax" "se"
        self.weight_share = True            #True
        self.warm_up = 0
        self.cell_express = ""              #""
        self.bi_optimize = ""              #"A_w" "A_A" 
        
        self.err_relative = False
        self.task = "train"
        #stat
        self.tX = 0

        if self.data_set=="cifar10":
            pass

    def problem(self):
        return self.data.problem()
        
    def legend(self):
        share = "" if self.weight_share else "***"
        attention = self.attention[0:3]
        express = self.cell_express
        leg = f"{express}{share}^{self.op_struc}^{self.primitive}^{attention}"
      
        return leg

    