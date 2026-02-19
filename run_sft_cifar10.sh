#!/bin/bash

# 2. Define Variables
METHOD="aft"
DATASET="cifar10"
#PRETRAINED="vit_giant_patch14_dinov2.lvd142m,vit_giant_patch14_clip_224.laion2b"
MODEL="vit_small_patch16_224.augreg_in1k"
LR="1e-4"
BETA="10"
SEED="0"

# 3. The Execution Command
python run.py \
--seed=${SEED} \
--model_class=${MODEL} \
--init_model=${MODEL} \
--dataset=${DATASET} \
--pretrained_models=None \
--train_frac=1 \
--use_val=False \
--method=${METHOD} \
--prec=${BETA} \
--learn_scales=True \
--steps=5000 \
--eval_steps=500 \
--optimizer=adam \
--batch_size=128 \
--lr=${LR} \
--wd=0 \
--no_augment=True \
--use_wandb=False