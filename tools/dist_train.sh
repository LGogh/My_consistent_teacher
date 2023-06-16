#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}



# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29506 tools/train.py configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p.py --launcher pytorch

# bash tools/dist_train.sh configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p.py 2

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29508 tools/train.py configs/baseline/mean_teacher_disambiguate_focal_180k_10p_2x8_fp16.py --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 tools/train.py configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p_2x8_fp16.py --launcher pytorch

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29509 tools/train.py configs/baseline/high_recall_v0.py --launcher pytorch

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29509 tools/train.py configs/myconfig/high_recall.py --launcher pytorch



# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 tools/train.py configs/myconfig/my_mean_teacher_retinanet_r50_fpn_voc0712_72k.py --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 tools/train.py configs/myconfig/test_v6_10p_2x2.py --launcher pytorch