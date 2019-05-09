# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import torch
from torch.autograd import Variable

sys.path.append("..")
from data.factory import detection_collate
from data.config import cfg
from s3fd import build_s3fd
from metric_val import VOC07MApMetric
from data.widerface import WIDERDetection
from torch.utils import data

parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str, default=os.path.join("..", 'weights/s3fd.pth'), help='trained model')
parser.add_argument('--thresh', default=0.005, type=float, help='Final confidence threshold')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if __name__ == '__main__':
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        # cudnn.benckmark = True
    save_path = 'eval_tools/s3fd_{}'.format("val")
    val_metric = VOC07MApMetric(ovp_thresh=0.5, class_names=['face'], roc_output_path=save_path)

    val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    val_loader = data.DataLoader(val_dataset, 4, num_workers=4, shuffle=False, collate_fn=detection_collate, pin_memory=True)

    for batch_idx, (images, targets) in enumerate(val_loader):
        images = Variable(images.cuda())

        detections = net(images)
        detections = detections.data
        detections = detections.cpu().numpy()
        det = detections[:, 1, :, :]

        val_metric.update(labels=targets, preds=det, thresh=args.thresh)

        if batch_idx % 10 == 0:
            print(batch_idx)

    names, values = val_metric.summary()
    for name, value in zip(names, values):
        print('Validation-{}={}'.format(name, value))
