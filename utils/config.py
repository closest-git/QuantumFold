'''
@Author: Yingshi Chen
@Date: 2020-02-14 11:59:10
@LastEditTime: 2020-04-24 15:36:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \QuantumForest\python-package\quantum_forest\QForest.py
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

        self.op_struc = "darts"           #"PCC" "darts" "pair"
        self.weights = "cys"              #"cys"  
        self.primitive = "p1"              #"p0"    "p1"    "c0"
        self.attention = "softmax"         #"entmax" 
        self.weight_share = True          #True
        self.warm_up = 50
        
        self.err_relative = False
        self.task = "train"
        #stat
        self.tX = 0

        if self.data_set=="cifar10":
            pass

    def problem(self):
        return self.data.problem()
        
    def model_info(self):
        return "QF_shallow"

    def env_title(self):
        title=f"{self.support.value}"
        if self.isFC:       title += "[FC]"
        if self.custom_legend is not None:
            title = title + f"_{self.custom_legend}"
        return title

    def main_str(self):
        main_str = f"{self.data_set}_ layers={self.nLayer} depth={self.depth} batch={self.batch_size} nTree={self.nTree} response_dim={self.response_dim} " \
            f"\nmax_out={self.max_out} choice=[{self.choice_func}] feat_info={self.feat_info}" \
            f"\nNO_ATTENTION={self.no_attention} reg_L1={self.reg_L1} path_way={self.path_way}"
        #if self.isFC:       main_str+=" [FC]"
        if self.custom_legend is not None:
            main_str = main_str + f"_{self.custom_legend}"
        return main_str