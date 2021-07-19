import sys

sys.path.append('..\\')

import argparse
import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
from core.utils import utils
from core.utils import warp_utils
from core.raft import RAFT
from core.datasets import FlyingChairs
from mydatasets.flyingchairsdata import flying_chairs
from core.utils.utils import ArrayToTensor
import time

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
CHECK_FREQ = 2000
EPSILON = 0.001
Q = 0.5

def sequence_photoloss(flow_preds_for, flow_preds_bac, image1, image2, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds_for)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)

        warpedimg, occmap = warp_utils.flow_warp(image2, flow_preds_for[i], flow_preds_bac[i])
        
        occmap_rgb = torch.cat((occmap, occmap, occmap), 1)
  
        numerator = torch.sum(torch.pow(torch.abs(image1 - warpedimg) + EPSILON, Q) * (occmap_rgb))
        denominator = torch.sum(occmap_rgb)
        i_loss = numerator / denominator

        flow_loss += i_weight * i_loss

    return flow_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': [368, 496], 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
    
    #train_dataset = FlyingChairs(aug_params, split='training', root='E:\RAFT datasets\FlyingChairs\data')
    
    # Colab:
    train_dataset = FlyingChairs(aug_params, split='training', root='/content/FlyingChairs_release/data')
    

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def train(args):
    torch.cuda.init()
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    train_loader = fetch_dataloader(args)

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    while should_keep_training:

        for _, (image1, image2, _, _) in enumerate(train_loader):
            optimizer.zero_grad()

            image1 = image1.cuda()
            image2 = image2.cuda()

            #start = time.time()

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_for = model(image1, image2, iters=args.iters)
            flow_bac = model(image2, image1, iters=args.iters)


            loss = sequence_photoloss(flow_for, flow_bac, image1, image2, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            #end = time.time()
            #print(end - start)

            if total_steps % 10 == 0:
                print('step %d, loss: %f' % (total_steps, loss))

            if total_steps % CHECK_FREQ == CHECK_FREQ - 1:
                PATH = '/checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                print('checkpoint saved !')

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break


    PATH = '/checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)