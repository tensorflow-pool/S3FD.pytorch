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
from data.widerface import detection_collate
from data.config import cfg
from s3fd import build_s3fd
from metric_val import VOC07MApMetric
from data.widerface import WIDERDetectionMat
from torch.utils import data

parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str, default=os.path.join("..", 'model/s3fd.pth'), help='trained model')
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
    val_metric = VOC07MApMetric(ovp_thresh=0.5, roc_output_path=save_path)
    curr_path = os.path.abspath(os.path.dirname(__file__))

    val_path = os.path.join(curr_path, "../eval_tools/ground_truth/wider_easy_val.mat")
    # val_path = os.path.join(curr_path, "../eval_tools/ground_truth/wider_medium_val.mat")
    # val_path = os.path.join(curr_path, "../eval_tools/ground_truth/wider_hard_val.mat")

    val_dataset = WIDERDetectionMat("/home/lijc08/datasets/widerface/WIDER_val/images", val_path, mode='val')
    val_loader = data.DataLoader(val_dataset, 1, num_workers=0, shuffle=False, collate_fn=detection_collate)
    img_count = len(val_dataset)
    for batch_idx, (images, targets, files) in enumerate(val_loader):
        images = Variable(images.cuda())

        detections = net(images)
        detections = detections.data
        detections = detections.cpu().numpy()
        det = detections[:, 1, :, :]

        val_metric.update(labels=targets, preds=det, files=files, thresh=args.thresh)

        if batch_idx % 10 == 0:
            print(batch_idx, img_count)

    names, values = val_metric.summary()
    for name, value in zip(names, values):
        print('Validation-{}={}'.format(name, value))
