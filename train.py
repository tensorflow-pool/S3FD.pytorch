# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable

from data.config import cfg
from data.factory import dataset_factory, detection_collate
from layers.modules import MultiBoxLoss
from s3fd import build_s3fd


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset',
                    default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--basenet',
                    default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default="model/s3fd.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Use mutil Gpu training')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

train_dataset, val_dataset = dataset_factory(args.dataset)

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
start_epoch = 0
s3fd_net = build_s3fd('train', cfg.NUM_CLASSES)
net = s3fd_net

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    epo = int(95000 / (len(train_dataset) / args.batch_size))
    start_epoch = net.load_weights(args.resume, epo)

else:
    vgg_weights = torch.load("weights/" + args.basenet)
    print('Load base network....')
    net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    if args.multigpu:
        net = torch.nn.DataParallel(s3fd_net)
    net = net.cuda()
    cudnn.benckmark = True

if not args.resume:
    print('Initializing weights...')
    s3fd_net.extras.apply(s3fd_net.weights_init)
    s3fd_net.loc.apply(s3fd_net.weights_init)
    s3fd_net.conf.apply(s3fd_net.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)
print('Loading wider dataset...')

save_folder = None

writer = None


def train():
    global save_folder
    global writer
    prefix = time.strftime("%Y-%m-%d-%H:%M:%S")
    save_folder = "train/models_{}".format(prefix)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler("{}/train.log".format(save_folder))
    # create formatter#
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # add formatter to ch
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.info('Using the specified args:')
    logging.info(args)
    writer = SummaryWriter(save_folder)

    step_index = 0
    iteration = int(start_epoch * len(train_dataset) / args.batch_size)
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            writer.add_scalar('loss', loss.item(), iteration)

            if iteration % 2 == 0:
                tloss = losses / (batch_idx + 1)
                # logging.info('Timer: %.4f' % (t1 - t0))
                logging.info('epoch:' + repr(epoch) + ' || iter:' + repr(iteration) + ' || Loss:%.4f' % (loss) + 'lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))
                logging.info('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(loss_c.item(), loss_l.item()))

                # for name, param in net.named_parameters():
                #     writer.add_histogram("data." + name, param.clone().cpu().data.numpy(), iteration, bins=100)
                #     writer.add_histogram("grad." + name, param.grad.clone().cpu().data.numpy(), iteration, bins=100)

                # w1 = torch.empty(256, 512, 3, 3)
                # nn.init.xavier_uniform_(w1)
                # writer.add_histogram("init.uniform", w1.cpu().data.numpy(), iteration, bins=100)
                #
                # w2 = torch.empty(256, 512, 3, 3)
                # nn.init.xavier_normal_(w2)
                # writer.add_histogram("init.normal", w2.cpu().data.numpy(), iteration, bins=100)
                #
                # writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), iteration)

            if iteration != 0 and iteration % 1000 == 0:
                logging.info('Saving state, iter: %s', iteration)
                file = 's3fd_' + repr(iteration) + '.pth'
                torch.save(s3fd_net.state_dict(), os.path.join(save_folder, file))
            iteration += 1

        # val(epoch)
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        step += 1

    tloss = (loc_loss + conf_loss) / step
    t2 = time.time()
    logging.info('Timer: %.4f' % (t2 - t1))
    logging.info('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        logging.info('Saving best state,epoch %s', epoch)
        file = 's3fd_{}.pth'.format(args.dataset)
        torch.save(s3fd_net.state_dict(), os.path.join(save_folder, file))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': s3fd_net.state_dict(),
    }
    file = 's3fd_{}_checkpoint.pth'.format(args.dataset)
    torch.save(states, os.path.join(save_folder, file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
