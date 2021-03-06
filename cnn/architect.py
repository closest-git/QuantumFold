'''
@Author: Yingshi Chen

@Date: 2020-04-14 10:32:20
@
# Description: 
'''
import torch
import numpy as np
import torch.nn as nn
import time
from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.abspath("utils"))
from config import *
from some_utils import *

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    
    print(f"\n======Architect parameters=")
    for i,param in enumerate(self.model.arch_parameters()):
      print(f"\t{i} {param.shape}",end="")
      #print(f"\t{i} {param}")
    #dump_model_params(self.model.arch_parameters())
    print(f"======"*16)
    print("")
  
  def __repr__(self):
    return f"======\nmomentum={self.network_momentum}\noptim={self.optimizer}"

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model
  
  #data-aware init
  def init_on_data(self,dataloader,criterion):
    config = self.model.config
    if config.op_struc != "se":    
      return
    config.search_alpha = True
    dataloader_iterator = iter(dataloader)
    try:
        data, target = next(dataloader_iterator)
        with torch.no_grad():
            result = self.model(data.cuda())    #参见_loss(self, input, target):
            loss = criterion(result, target.cuda())
        
        for ATT_weight in self.model.listWeight:
          ATT_weight.step()
          if ATT_weight.isReduce:     #仅用于兼容darts
              self.model.alphas_reduce = ATT_weight.alphas_
          else:
              self.model.alphas_normal = ATT_weight.alphas_
        print(f"====== Architect::init_on_data\tdata={data.shape},loss={loss.item()}")
        print(f"")
        config.search_alpha=False
    except StopIteration:
        return

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    #t0=time.time()
    self.model.config.search_alpha=True
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

    for ATT_weight in self.model.listWeight:
      ATT_weight.step()
      if ATT_weight.isReduce:     #仅用于兼容darts
          self.model.alphas_reduce = ATT_weight.alphas_
      else:
          self.model.alphas_normal = ATT_weight.alphas_
    self.model.config.search_alpha=False
    #print(f"Architect::step T={time.time()-t0:.3f}")

  '''
    The search procedure stops when there are two or more than two skip-connects in one cell.
  '''
  def isEarlyStopping(self):
    for cell in self.model.cells:
      pass
    return False

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

