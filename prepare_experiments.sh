#!/bin/bash

# ========== КОНФИГУРАЦИЯ ==========
MODEL="vit_small_patch16_224.augreg_in1k"        # ViT-S ~22M
TEACHER="vit_base_patch14_dinov2.lvd142m"        # Ваш teacher
SEEDS=(0 1 2)
BETA=10
STEPS=5000
LR=1e-4
BATCH=128

# Датасеты для разнообразия
DATASETS=("flowers")

# Создаем папку для фичей, если она не существует
mkdir -p ./features

echo "=== FEATURES COMPUTATION ==="
for DS in "${DATASETS[@]}"; do
    echo "Dataset: $DS"
    
    # Teacher features
    SAVE_T="./features/${TEACHER}_${DS}.pt"
    python save_features.py --model_class="$TEACHER" --dataset="$DS" --save_path="$SAVE_T"
    
    # Downstream features (для LogME)
    SAVE_M="./features/${MODEL}_${DS}.pt"
    python save_features.py --model_class="$MODEL" --dataset="$DS" --save_path="$SAVE_M"
done

DATASETS=("cifar100" "cifar10" "flowers")

echo "=== LogME for B-Tuning ==="
for DS in "${DATASETS[@]}"; do
    # Проходим по списку моделей
    for m in "$MODEL" "$TEACHER"; do
        python LogME.py --model_class="$m" --dataset="$DS"
    done
done