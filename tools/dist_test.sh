#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

# CONFIG=configs/baseline/mean_teacher_disambiguate_focal_180k_10p_2x8_fp16.py
# CHECKPOINT=work_dirs/mean_teacher_disambiguate_focal_180k_10p_2x8_fp16/iter_8000.pth

CONFIG=configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p_2x8_fp16.py
CHECKPOINT=work_dirs/mean_teacher_retinanet_r50_fpn_coco_180k_10p_2x8_fp16/180*4/iter_720000.pth

CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29504 \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --eval bbox \
    --cfg-options model.test_cfg.score_thr=0.001 \
    # --out output/coco_4k_lr00251.pkl \