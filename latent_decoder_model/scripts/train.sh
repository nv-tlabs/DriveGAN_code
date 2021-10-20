#!/usr/bin/env bash

src_dir=$1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=6003 main.py \
    --path ${src_dir} \
    --batch 6 \
    --size 256 \
    --dataset carla \
    --gamma 50.0 \
    --theme_beta 1.0 \
    --spatial_beta 2.0 \
    --log_dir ./logs/vaegan
