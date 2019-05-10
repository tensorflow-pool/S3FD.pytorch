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

import cv2
import numpy as np

sys.path.append("..")
from data.config import cfg


def parse_file(root, file):
    img_paths = []
    img_faces = []
    index = 0
    with open(file, 'r') as gtfile:
        while (True):  # and len(faces)<10
            rel_imgpath = gtfile.readline()[:-1]
            if (rel_imgpath == ""):
                break;
            if index % 100 == 0:
                print(index)
            file_path = os.path.join(root, rel_imgpath + ".jpg")
            img = cv2.imread(file_path)
            numbbox = int(gtfile.readline())
            bboxes = []
            for i in range(numbbox):
                line = gtfile.readline()
                line = line.split()

                major_axis_radius = (float)(line[0])
                minor_axis_radius = (float)(line[1])
                angle = (float)(line[2])
                center_x = (float)(line[3])
                center_y = (float)(line[4])
                score = (float)(line[5])

                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)), ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                r = cv2.boundingRect(mask)
                data = [r[0], r[1], r[2], r[3]]
                bboxes.append(data)
            # for box
            img_paths.append(file_path)
            img_faces.append(bboxes)
            index += 1
    return img_paths, img_faces


def trans_format():
    FDDB_ROOT = cfg.FACE.FDDB_DIR
    if os.path.exists(cfg.FACE.FDDB_VAL_FILE):
        os.remove(cfg.FACE.FDDB_VAL_FILE)

    fw = open(cfg.FACE.FDDB_VAL_FILE, 'w')
    for i in range(1, 11):
        origin_file = os.path.join(FDDB_ROOT, "FDDB-folds", "FDDB-fold-%02d-ellipseList.txt" % i)
        img_paths, bbox = parse_file(os.path.join(FDDB_ROOT, "originalPics"), origin_file)
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


if __name__ == '__main__':
    trans_format()
