# -*- coding:utf-8 -*-
'''
- [ ] voc格式：
xml

- [ ] standard格式1：box为int的ltwh
path count, [box, label]*

- [ ] standard格式2：box为int的ltwh
path
count
box extras

- [ ] fddb
path
count
ellipse label

- [ ] mxnet格式：box为float的ltrb
index，2， item_count=6, [label, box,  extra]*, path
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append("..")
from data.config import cfg


def parse_file(img_root, list_file):
    with open(list_file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(img_root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in range(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def trans_dataset(img_root, list_file, target_file):
    img_paths, bbox = parse_file(img_root, list_file)
    fw = open(target_file, 'w')
    for index in range(len(img_paths)):
        path = img_paths[index]
        boxes = bbox[index]
        fw.write(path)
        fw.write(' {}'.format(len(boxes)))
        for box in boxes:
            data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
            fw.write(data)
        fw.write('\n')
    fw.close()


def trans_format():
    WIDER_ROOT = cfg.WIDER_DIR

    train_img_root = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
    train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    target_file = cfg.WIDER_TRAIN_FILE
    trans_dataset(train_img_root, train_list_file, target_file)

    val_img_root = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')
    val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    target_file = cfg.WIDER_VAL_FILE
    trans_dataset(val_img_root, val_list_file, target_file)


if __name__ == '__main__':
    trans_format()
