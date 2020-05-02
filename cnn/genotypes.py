'''
@Author: Yingshi Chen

@Date: 2020-04-14 10:32:20
@
# Description: 
'''
from collections import namedtuple
import torch.nn.functional as F
import torch
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES_darts = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_cifar = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


PCDARTS = PC_DARTS_cifar

S_CYS_cifar = Genotype(
    normal=[('DepthConv_3', 1), ('DepthConv_3', 0), ('DepthConv_3', 1), ('Conv_3', 2), ('Conv_3', 3), ('Conv_3', 2), ('DepthConv_3', 0), ('DepthConv_3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('Conv_3', 2), ('max_pool_3x3', 0), ('Conv_3', 3), ('Conv_3', 2), ('Conv_3', 4), ('Conv_3', 3)], reduce_concat=range(2, 6))

G_C_se = Genotype(
    normal=[('DepthConv_3', 0), ('DepthConv_3', 1), ('ReLU', 1), ('Conv_3', 0), ('DepthConv_3', 3), ('DepthConv_3', 1), ('Conv_3', 4), ('DepthConv_3', 1)],normal_concat=[2, 3, 4, 5], 
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('BatchNorm2d', 2), ('max_pool_3x3', 0), ('BatchNorm2d', 2), ('DepthConv_3', 3), ('Conv_3', 4), ('Identity', 2)], reduce_concat=[2, 3, 4, 5])

def dump_seperate_genotype(model, logging):
    for id, cell in enumerate(model.cells):
        gene = cell.weight2gene()
        print(f"cell_{id}\t{gene}")    

def dump_genotype(model, logging):
    print("=================="*6)
    if not model.config.weight_share:
        dump_seperate_genotype(model,logging)
        return

    PRIMITIVES_pool = model.config.PRIMITIVES_pool
    genotype,isValid = model.genotype()
    if not isValid:
        return

    logging.info('genotype = %s', genotype)
    genotype_1 = model.cells[1].weight2gene()
    if genotype_1 not in genotype:
        print(f"\n!!!GENOTYPE MisMatch!!! \n{genotype_1}\n{genotype}\n!!!GENOTYPE MisMatch!!!\n")
    if True:
        alphas_normal = model.cells[1].get_alpha()
    else:
        alphas_normal = model._arch_parameters[0]
        alphas_normal = F.softmax(alphas_normal, dim=-1).detach().cpu().numpy()
    nRow, nCol = alphas_normal.shape
    assert nRow==14 and nCol==8
    for r in range(nRow):
        ids = sorted(range(nCol), key=lambda c: -alphas_normal[r, c])
        w0 = alphas_normal[r, ids[0]]
        for c in ids:
            if w0 == alphas_normal[r, c]:
                print(f"{PRIMITIVES_pool[c]}\t", end="")
            else:
                print(
                    f"{PRIMITIVES_pool[c]}-{w0-alphas_normal[r,c]:.3f} ", end="")
        print("")
    # print(alphas_normal)
    if False:
        values, indices = torch.max(alphas_normal, 1)
        for val, typ in zip(values, indices):
            PRIMITIVE = model.config.PRIMITIVES_pool[typ.item()]
            print(f"\"{PRIMITIVE}\"={val.item():.4f},", end="")
    if not model.config.weight_share:
        for id, cell in enumerate(model.cells):
            gene = cell.weight2gene()
            print(f"cell_{id}\t{gene}")

    print("=================="*6)
