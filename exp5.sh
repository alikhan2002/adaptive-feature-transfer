#!/bin/bash
set -euo pipefail

# Downstream model (student)
MODEL="vit_small_patch16_224.augreg_in1k"

# Teacher (features must exist per dataset)
TEACHER="vit_base_patch14_dinov2.lvd142m"

# Shared training hyperparams (как в README example)
BETA=10
STEPS=5000
LR="1e-4"
BATCH=128
SEED=0

mkdir -p results features

for DS in "pets"; do #"cifar10"
  echo "=== EXP5 ${DS} ==="

  # For KD/AFT we need precomputed teacher features saved as ./features/{TEACHER}_{DS}.pt [web:15]
  FEAT_PATH="./features/${TEACHER}_${DS}.pt"
  if [ ! -f "$FEAT_PATH" ]; then
    echo "[EXP5] Missing features: $FEAT_PATH"
    echo "[EXP5] Computing features with save_features.py ..."
    python save_features.py \
      --model_class "${TEACHER}" \
      --dataset "${DS}" \
      --save_path "${FEAT_PATH}"
  fi

  # 1) init = standard transfer learning baseline [web:15]
  python run.py \
    --seed "${SEED}" \
    --model_class "${MODEL}" \
    --init_model "${MODEL}" \
    --dataset "${DS}" \
    --pretrained_models "none" \
    --train_frac 1 \
    --use_val False \
    --method init \
    --prec 0 \
    --steps "${STEPS}" \
    --eval_steps 500 \
    --optimizer adam \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --wd 0 \
    --no_augment True \
    --use_wandb False \
    | tee "results/exp5_${DS}_init.log"

  # 2) kd baseline (uses teacher features) [web:15]
  python run.py \
    --seed "${SEED}" \
    --model_class "${MODEL}" \
    --init_model "${MODEL}" \
    --dataset "${DS}" \
    --pretrained_models "${TEACHER}" \
    --train_frac 1 \
    --use_val False \
    --method kd \
    --prec "${BETA}" \
    --steps "${STEPS}" \
    --eval_steps 500 \
    --optimizer adam \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --wd 0 \
    --no_augment True \
    --use_wandb False \
    | tee "results/exp5_${DS}_kd.log"

  # 3) aft (uses teacher features) [web:15]
  python run.py \
    --seed "${SEED}" \
    --model_class "${MODEL}" \
    --init_model "${MODEL}" \
    --dataset "${DS}" \
    --pretrained_models "${TEACHER}" \
    --train_frac 1 \
    --use_val False \
    --method aft \
    --prec "${BETA}" \
    --learn_scales True \
    --steps "${STEPS}" \
    --eval_steps 500 \
    --optimizer adam \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --wd 0 \
    --no_augment True \
    --use_wandb False \
    | tee "results/exp5_${DS}_aft.log"
done

echo "Done. Parse:"
echo "grep -h 'Final test acc' results/exp5_*.log | sort"
