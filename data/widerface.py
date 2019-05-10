# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from PIL import Image

from utils.augmentations import preprocess


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w, image_path = self.pull_item(index)
        return img, target, image_path

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack((sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, self.num_samples)

        # img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), target, im_height, im_width, image_path

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


class WIDERDetectionMat(WIDERDetection):
    """docstring for WIDERDetection"""

    def __init__(self, root, mat_file, mode='val'):
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        mat_datas = sio.loadmat(mat_file)

        event_list = mat_datas["event_list"]
        gt_list = mat_datas["gt_list"]
        face_bbx_list = mat_datas["face_bbx_list"]
        file_list = mat_datas["file_list"]
        event_cont = len(event_list)
        for event_index in range(event_cont):
            event = event_list[event_index][0][0]
            gts = gt_list[event_index][0]
            face_bbxs = face_bbx_list[event_index][0]
            files = file_list[event_index][0]
            files_count = len(files)
            for file_index in range(files_count):
                gt_indexes = gts[file_index][0]
                bbxs = face_bbxs[file_index][0]
                file = files[file_index][0][0]
                box = []
                label = []
                num_faces = len(bbxs)
                for index in gt_indexes:
                    i = index[0] - 1
                    x = float(bbxs[i][0])
                    y = float(bbxs[i][1])
                    w = float(bbxs[i][2])
                    h = float(bbxs[i][3])
                    if w <= 0 or h <= 0:
                        continue
                    box.append([x, y, x + w, y + h])
                    label.append(1)
                if len(box) > 0:
                    full_path = os.path.join(root, event, file + ".jpg")
                    if os.path.exists(full_path):
                        self.fnames.append(full_path)
                        self.boxes.append(box)
                        self.labels.append(label)
                    else:
                        print("miss ", full_path)
        self.num_samples = len(self.boxes)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    files = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        files.append(sample[2])
    return torch.stack(imgs, 0), targets, files


if __name__ == '__main__':
    dataset = WIDERDetectionMat("/home/lijc08/datasets/widerface/WIDER_val/images", "../eval_tools/ground_truth/wider_easy_val.mat")
    # for i in range(len(dataset)):
    img, target, im_height, im_width = dataset.pull_item(14)
    print(img, target, im_height, im_width)
