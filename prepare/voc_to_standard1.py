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

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

keep_difficult = True


def parse_file(img_root, file_list, box_root):
    img_paths = []
    img_boxes = []
    for line in open(file_list):
        img_path = os.path.join(img_root, line.strip() + ".jpg")

        box_path = os.path.join(box_root, line.strip() + ".xml")
        target = ET.parse(box_path).getroot()
        boxes = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            xmin = int(bbox.find("xmin").text)
            xmax = int(bbox.find("xmax").text)
            ymin = int(bbox.find("ymin").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

        img_paths.append(img_path)
        img_boxes.append(boxes)
    return img_paths, img_boxes


def trans_dataset(img_root, file_list, box_root, target_file):
    img_paths, bbox = parse_file(img_root, file_list, box_root)
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


def trans_datasets(img_roots, file_lists, box_roots, target_file):
    fw = open(target_file, 'w')
    for img_root, file_list, box_root in zip(img_roots, file_lists, box_roots):
        img_paths, bbox = parse_file(img_root, file_list, box_root)
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


def get_config(dataset_root, list_file_name, sub_datasets):
    img_roots, list_files, box_roots = [], [], []
    for sub in sub_datasets:
        img_roots += [os.path.join(dataset_root, '{}/JPEGImages'.format(sub))]
        list_files += [os.path.join(dataset_root, '{}/ImageSets/Main/{}'.format(sub, list_file_name))]
        box_roots += [os.path.join(dataset_root, '{}/Annotations'.format(sub))]
    return img_roots, list_files, box_roots


def trans_format():
    # HEAD
    img_roots, list_files, box_roots = get_config(cfg.HEAD_DIR, "train.txt", ["SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"])
    target_file = cfg.HEAD_TRAIN_FILE
    trans_datasets(img_roots, list_files, box_roots, target_file)

    img_roots, list_files, box_roots = get_config(cfg.HEAD_DIR, "val.txt", ["SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"])
    target_file = cfg.HEAD_VAL_FILE
    trans_datasets(img_roots, list_files, box_roots, target_file)

    img_roots, list_files, box_roots = get_config(cfg.HEAD_DIR, "test.txt", ["SCUT_HEAD_Part_A", "SCUT_HEAD_Part_B"])
    target_file = cfg.HEAD_TEST_FILE
    trans_datasets(img_roots, list_files, box_roots, target_file)


if __name__ == '__main__':
    trans_format()
