#!/usr/bin/env bash

import os
os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"

WANDB_API_KEY=39dbffbe34bf010c597b2cbf432b89ae7060d03e
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 tools/train.py configs/myconfig/test_v6_10p_2x2.py --launcher pytorch