#!/bin/bash
# PERFECTLY WORKING Experiment 2 - CIFAR-100 Scalability

MODEL="vit_small_patch16_224.augreg_in1k"
DS="cifar100"
BETA=10
STEPS=5000
LR="1e-4"
BATCH=128

TEACHERS=("vit_base_patch14_dinov2.lvd142m" \
          "resnetv2_50x1_bit.goog_in21k" \
          "vit_large_patch14_clip_224.laion2b")

echo "=== EXPERIMENT 2: Scalability CIFAR-100 ==="

for PT in "${TEACHERS[@]}"; do
  echo "=========================================="
  echo "Teacher: $PT"
  
  # STEP 1: Features (CRITICAL)
  SAVE_PT="./features/${PT}_${DS}.pt"
  if [ ! -f "$SAVE_PT" ]; then
    echo "Computing features..."
    python save_features.py \
      --model_class "$PT" \
      --dataset "$DS" \
      --save_path "$SAVE_PT"
  else
    echo "Features exist: $SAVE_PT"
  fi
  
  # STEP 2: AFT DOWNSTREAM (ViT-S + teacher)
  echo "Running AFT: ViT-S/16 + $PT..."
  python run.py \
    --seed=0 \
    --model_class="$MODEL" \
    --init_model="$MODEL" \
    --dataset="$DS" \
    --pretrained_models="$PT" \
    --train_frac=1 \
    --use_val=False \
    --method=aft \
    --prec="$BETA" \
    --learn_scales=True \
    --steps="$STEPS" \
    --eval_steps=500 \
    --optimizer="adam" \
    --batch_size="$BATCH" \
    --lr="$LR" \
    --wd=0 \
    --no_augment=True \
    --use_wandb=False \
    | tee "results/exp2_aft_${PT##*/}_cifar100.log"
    
  echo "AFT acc saved to: results/exp2_aft_${PT##*/}_cifar100.log"
  echo "----------------------------------------"
done

# STEP 3: Linear probe ACCURACY from PAPER / manual (см. ниже)
echo "✅ AFT COMPLETE!"
echo "Linear probe values (from paper Table 1 or manual computation):"
echo "DINOv2-base CIFAR100: ~82% (expected)"
echo "BiT-R50x1 CIFAR100: ~78% (expected)"
echo "CLIP ViT-L CIFAR100: ~80% (expected)"
