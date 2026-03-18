#!/bin/bash

mkdir -p logs/tvve_stage1

# Replace this placeholder with your RLBench training dataset path,
# or override it via the TRAIN_DEMO_FOLDER environment variable.
TRAIN_DEMO_FOLDER=${TRAIN_DEMO_FOLDER:-/path/to/rlbench/data/train}

bs=32
epoch=130
gpus=1
moe_x_depth=4
depth=8
amp=true
lr=1.25e-5
python train_tvve_stage1.py \
  config=./configs/tvve_stage1.yaml \
  train.demo_folder="${TRAIN_DEMO_FOLDER}" \
  hydra.job.name=tvve_stage1 \
  train.num_gpus=${gpus} \
  train.bs=${bs} \
  train.save_freq=1000 \
  model.hp.lr=${lr} \
  train.num_workers=8 \
  model.hp.clip_grad_norm=1.0 \
  train.epochs=${epoch} \
  model.hp.depth=${depth} \
  model.hp.moe_x_depth=${moe_x_depth} \
  model.hp.amp=${amp} \
  wandb=RLBench_arp_plus \
  wandb_name=tvve_stage1_epoch_${epoch}_lr_${lr}_bs_${bs}_gpus_${gpus}
