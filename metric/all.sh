#!/usr/bin/env bash
MODEL_PATH=../model/sfd_face_169000.pth
MODEL_PATH=../model/s3fd.pth

#python standard_pr_roc.py --thresh=0.5 --model=$MODEL_PATH --standard_file=~/datasets/fddb/fddb_val_standard1.txt
#python standard_pr_roc.py --thresh=0.05 --model=$MODEL_PATH --standard_file=~/datasets/fddb/fddb_val_standard1.txt

#python standard_pr_roc.py --thresh=0.5 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_train_standard1.txt
#python standard_pr_roc.py --thresh=0.05 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_train_standard1.txt
#
#python standard_pr_roc.py --thresh=0.5 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_val_standard1.txt
#python standard_pr_roc.py --thresh=0.05 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_val_standard1.txt
#
#python standard_pr_roc.py --thresh=0.5 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_test_standard1.txt
python standard_pr_roc.py --thresh=0.05 --model=$MODEL_PATH --standard_file=~/datasets/SCUT_HEAD/head_test_standard1.txt


#python wider_pr_roc.py --dataset=easy --thresh=0.05 --model=$MODEL_PATH
#python wider_pr_roc.py --dataset=easy --thresh=0.5 --model=$MODEL_PATH

#python wider_pr_roc.py --dataset=medium --thresh=0.05 --model=$MODEL_PATH
#python wider_pr_roc.py --dataset=medium --thresh=0.5 --model=$MODEL_PATH

#python wider_pr_roc.py --dataset=hard --thresh=0.05 --model=$MODEL_PATH
#python wider_pr_roc.py --dataset=hard --thresh=0.5 --model=$MODEL_PATH

