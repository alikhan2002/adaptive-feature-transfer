#!/bin/bash
set -euo pipefail

MODEL="vit_small_patch16_224.augreg_in1k"
TEACHER="vit_base_patch14_dinov2.lvd142m"
DS="cifar100"

BETA=10
STEPS=5000
LR="1e-4"
BATCH=128
SEED=0

mkdir -p results features

echo "=== EXP3: Noise robustness (CIFAR-100) ==="

# sanity check: base teacher features must exist
BASE_FEAT="./features/${TEACHER}_${DS}.pt"
if [ ! -f "$BASE_FEAT" ]; then
  echo "Base features not found: $BASE_FEAT"
  echo "Run first: python save_features.py --model_class ${TEACHER} --dataset ${DS} --save_path ${BASE_FEAT}"
  exit 1
fi

NOISE_DIMS=(0 512 2048)

for dim in "${NOISE_DIMS[@]}"; do
  echo "------------------------------------------"
  echo "Noise dim = $dim"

  if [ "$dim" -eq 0 ]; then
    PT_MODELS="$TEACHER"
  else
    NOISY_MODEL="${TEACHER}_noise${dim}"
    NOISY_PATH="./features/${NOISY_MODEL}_${DS}.pt"   # IMPORTANT naming: {model}_{dataset}.pt

    python create_noise_features.py \
      --base_path "$BASE_FEAT" \
      --noise_dim "$dim" \
      --save_path "$NOISY_PATH" \
      --seed "$SEED"

    PT_MODELS="$NOISY_MODEL"
  fi

  # KD
  python run.py \
    --seed "$SEED" \
    --model_class "$MODEL" \
    --init_model "$MODEL" \
    --dataset "$DS" \
    --pretrained_models "$PT_MODELS" \
    --train_frac 1 \
    --use_val False \
    --method kd \
    --prec "$BETA" \
    --steps "$STEPS" \
    --eval_steps 500 \
    --optimizer adam \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --wd 0 \
    --no_augment True \
    --use_wandb False \
    | tee "results/exp3_kd_noise${dim}.log"

  # AFT
  python run.py \
    --seed "$SEED" \
    --model_class "$MODEL" \
    --init_model "$MODEL" \
    --dataset "$DS" \
    --pretrained_models "$PT_MODELS" \
    --train_frac 1 \
    --use_val False \
    --method aft \
    --prec "$BETA" \
    --learn_scales True \
    --steps "$STEPS" \
    --eval_steps 500 \
    --optimizer adam \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --wd 0 \
    --no_augment True \
    --use_wandb False \
    | tee "results/exp3_aft_noise${dim}.log"
done

echo "✅ EXP3 done."
echo "Parse with: grep -h 'Final test acc' results/exp3_*.log"
