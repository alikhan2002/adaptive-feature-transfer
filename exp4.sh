#!/bin/bash
set -euo pipefail

MODEL="vit_small_patch16_224.augreg_in1k"
TEACHER="vit_base_patch14_dinov2.lvd142m"
DS="cifar100"
SEED=0

BETA=10
STEPS=5000
LR="1e-4"
BATCH=128

mkdir -p results

echo "=== EXP4 (minimal, no code changes): CIFAR-100 ==="

# 1) Full AFT (kernel + learned diagonal μ)
python run.py \
  --seed $SEED \
  --model_class $MODEL --init_model $MODEL \
  --dataset $DS --pretrained_models $TEACHER \
  --train_frac 1 --use_val False \
  --method aft --prec $BETA --learn_scales True \
  --steps $STEPS --eval_steps 500 \
  --optimizer adam --batch_size $BATCH --lr $LR --wd 0 \
  --no_augment True --use_wandb False \
  | tee results/exp4_full_aft.log

# 2) Identity μ (kernel + fixed μ)
python run.py \
  --seed $SEED \
  --model_class $MODEL --init_model $MODEL \
  --dataset $DS --pretrained_models $TEACHER \
  --train_frac 1 --use_val False \
  --method aft --prec $BETA --learn_scales False \
  --steps $STEPS --eval_steps 500 \
  --optimizer adam --batch_size $BATCH --lr $LR --wd 0 \
  --no_augment True --use_wandb False \
  | tee results/exp4_identity_mu.log

# 3) No regularization (β=0) — baseline
python run.py \
  --seed $SEED \
  --model_class $MODEL --init_model $MODEL \
  --dataset $DS --pretrained_models $TEACHER \
  --train_frac 1 --use_val False \
  --method aft --prec 0 --learn_scales True \
  --steps $STEPS --eval_steps 500 \
  --optimizer adam --batch_size $BATCH --lr $LR --wd 0 \
  --no_augment True --use_wandb False \
  | tee results/exp4_no_reg.log

echo "Parse:"
echo "grep -h 'Final test acc' results/exp4_*.log"
