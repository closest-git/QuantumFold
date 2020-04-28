
import os
import sys
import time
import glob
import numpy as np
np.set_printoptions(linewidth=np.inf)
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
#from cifar_x import *
import torch.backends.cudnn as cudnn
sys.path.append(os.path.abspath("utils"))
# print(sys.path)
from config import *
from some_utils import *
from architect import Architect
from model_search import Network
from torch.autograd import Variable
from Visualizing import *
from genotypes import *
from experiment import *


'''
    python cnn/search_darts.py --gpu 1
'''

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10',help='location of the data corpus')
parser.add_argument('--batch_size', type=int,default=96, help='batch size')  # 256
parser.add_argument('--learning_rate', type=float,default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,help='num of training epochs')
parser.add_argument('--init_channels', type=int,default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,help='total number of layers')
parser.add_argument('--model_path', type=str,default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,default=1e-3, help='weight decay for arch encoding')


if False:
    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save,
                                      time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10



def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    config = QuantumFold_config(None, 0)
    if config.op_struc == "":        args.batch_size = args.batch_size//4
    
    config.device = OnInitInstance(args.seed, args.gpu)
    if config.primitive == "p0":
        config.PRIMITIVES_pool = ['none','max_pool_3x3','avg_pool_3x3','Identity','BatchNorm2d','ReLU','Conv_3','Conv_5']
    elif config.primitive == "p1":
        config.PRIMITIVES_pool = ['none','max_pool_3x3','Identity','BatchNorm2d','ReLU','Conv_3','DepthConv_3','Conv_11']
    elif config.primitive == "p2":
        config.PRIMITIVES_pool = ['none','max_pool_3x3','Identity','BatchNorm2d','ReLU','Conv_3','DepthConv_3','Conv_11']
    elif config.primitive == "c0":       
        config.PRIMITIVES_pool = ['none','max_pool_3x3','avg_pool_3x3','skip_connect','sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5']
    args.load_workers = 8

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(config, args.init_channels,CIFAR_CLASSES, args.layers, criterion)
    print(model)
    # dump_model_params(model)
    model = model.cuda()
    model.visual =  Visdom_Visualizer(env_title=f"{args.set}_{model.title}")
    model.visual.img_dir = "./results/images/"
    logging.info("param size = %.3fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        #train_data = CIFAR10_x(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.load_workers) #args.load_workers

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True, num_workers=0)
    config.experiment = Experiment(config,"cifar_10",model,loss_fn=None,optimizer=optimizer,objective_metric=None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    print(architect)
    print(f"======\tconfig={config.__dict__}")
    print(f"======\targs={args.__dict__}")
    t0 = time.time()
    for epoch in range(args.epochs):      
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        dump_genotype(model,logging)

        # training
        train_acc, train_obj = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info(f'train_acc {train_acc} T={time.time()-t0:.2f}')

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        logging.info(f'valid_acc {valid_acc} T={time.time()-t0:.2f}')
        config.experiment.best_score = max(valid_acc,config.experiment.best_score)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        model.visual.UpdateLoss(title=f"Accuracy on \"{args.set}\"",legend=f"{model.title}", loss=valid_acc,yLabel="Accuracy")


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    tX,t0 = 0,time.time()
    best_prec = 0
    isArchitect = not model.config.experiment.isKeepWarm()     
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        # get a random minibatch from the search queue with replacement
        t1 = time.time()
        if isArchitect:     # two-stage training
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda()    
            architect.step(input, target, input_search, target_search,lr, optimizer, unrolled=args.unrolled)
        
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        tX += time.time()-t1
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        best_prec = max(best_prec,prec1.item())
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %.5f %.3f %.3f', step,objs.avg, top1.avg, top5.avg)
        #model.visual.UpdateLoss(title=f"Accuracy on train",legend=f"{model.title}", loss=prec1.item(),yLabel="Accuracy")
        print(f'\r\t{model.title}_{step}@{epoch}:\tloss={objs.avg:.3f}, top1={top1.avg:.2f}, top5={top5.avg:.2f} T={time.time()-t0:.1f}({tX:.3f})\t', end="")
        #break
    print(f'train_{epoch}:best_prec={best_prec}\tT={time.time()-t0:.3f}')

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,epoch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                #logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                print(f'\r\tvalid {step}@{epoch}:\t{objs.avg:.3f}, {top1.avg:.3f}, {top5.avg:.3f}', end="")
    print(f'\tinfer_:\tT={time.time()-t0:.3f}')

    return top1.avg, objs.avg


if __name__ == '__main__':
    args = parser.parse_args()
    args.save = './search/{}-{}'.format(args.save,time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    if args.set == 'cifar100':
        CIFAR_CLASSES = 100

    main()
