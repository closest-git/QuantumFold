import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'global_pool' : lambda C, stride, affine: nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(None,None)),Identity_Stride_(stride)
    ),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'ReLU' : lambda C, stride, affine: nn.Sequential(
      nn.ReLU(inplace=False),
      Identity_Stride_(stride)
    ),
  'Identity' : lambda C, stride, affine: Identity_Stride_(stride),
  'BatchNorm2d' : lambda C, stride, affine: nn.Sequential(
      nn.BatchNorm2d(C, affine=affine),
      Identity_Stride_(stride)
    ),
  'Conv_3' : lambda C, stride, affine: nn.Conv2d(C, C, 3, stride,padding=1,   bias=False),
  'Conv_5' : lambda C, stride, affine: nn.Conv2d(C, C, 5, stride,padding=2,   bias=False),
  'DepthConv_3' : lambda C, stride, affine: nn.Conv2d(C, C, 3, stride,padding=2, dilation=2, groups=C, bias=False),
  'DepthConv_5' : lambda C, stride, affine: nn.Conv2d(C, C, 5, stride,padding=4, dilation=2, groups=C, bias=False),
  'Conv_11' : lambda C, stride, affine: nn.Conv2d(C, C, kernel_size=1, stride=stride,padding=0, bias=False),
  
  
}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
    # if C_in==256:
    #     C_in=256
    self.desc = f"ReLUConvBN_C({C_in},{C_out})_K{kernel_size}_s{stride}_a({affine})"

  def forward(self, x):
    return self.op(x)
  
  def __repr__(self):
    return self.desc

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.desc = f"DilConv_d{dilation}_C({C_in},{C_out})_K{kernel_size}_s{stride}_a({affine})"

  def forward(self, x):
    return self.op(x)
  
  def __repr__(self):
    return self.desc

#'ConvY_5' : lambda C, stride, affine: nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.desc = f"SepConv_C({C_in},{C_out})_K{kernel_size}_s{stride}_a({affine})"

  def forward(self, x):
    return self.op(x)
  
  def __repr__(self):
    return self.desc


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Identity_Stride_(nn.Module):
  def __init__(self, stride):
    super(Identity_Stride_, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x
    return x[:,:,::self.stride,::self.stride]

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)




class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

