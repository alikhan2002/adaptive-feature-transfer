#!/bin/bash

MODEL="vit_small_patch16_224.augreg_in1k"        # ViT-S ~22M
TEACHER="vit_base_patch14_dinov2.lvd142m"        # Ваш teacher
SEEDS=(0 1 2)
BETA=10
STEPS=5000
LR=1e-4
BATCH=128

DS="cifar100"
PT_MODELS=$TEACHER

echo "=== EXP1 CIFAR100 ==="
for method in "init" "kd" "aft"; do
  echo "Method: $method"
  for seed in "${SEEDS[@]}"; do
    python run.py \
      --seed=$seed \
      --model_class=$MODEL --init_model=$MODEL \
      --dataset=$DS --pretrained_models=$PT_MODELS \
      --train_frac=1 --use_val=False \
      --method=$method \
      --prec=$BETA \
      --learn_scales=True \
      --steps=$STEPS --eval_steps=500 \
      --optimizer=adam --batch_size=$BATCH --lr=$LR \
      --wd=0 --no_augment=True --use_wandb=False \
      | tee "results/exp1_${method}_seed${seed}.log"
  done
done