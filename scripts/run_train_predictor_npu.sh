#!/usr/bin/env bash

set -Eeuo pipefail

# Run from the repository root. Only one physical NPU is exposed to MLP training.
export ASCEND_RT_VISIBLE_DEVICES=0

python Predictor0916/scripts/train_predictor_8shards.py \
  --preprocessed-dir Predictor0916/data/qwen3.6-preprocessed \
  --output-dir Predictor0916/outputs/qwen3.6-predictor \
  --expected-shards 8 \
  --device npu:0 \
  --num-bins 20 \
  --target-quantiles 0.01 0.99 \
  --loss-type soft_label \
  --lambda-val 0.95 \
  --epochs 10 \
  --batch-size 256 \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --patience 3 \
  --seed 42
