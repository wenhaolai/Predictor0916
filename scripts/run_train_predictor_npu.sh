#!/usr/bin/env bash

set -Eeuo pipefail

# Run from the repository root. Only one physical NPU is exposed to MLP training.
export ASCEND_RT_VISIBLE_DEVICES=0

python Predictor0916/scripts/train_predictor_8shards.py \
  --preprocessed-dir Predictor0916/data/qwen3.6-preprocessed \
  --output-dir Predictor0916/outputs/qwen3.6-predictor \
  --shard-pattern "shard-*.csv" \
  --expected-shards 8 \
  --device npu:0 \
  --num-bins 20 \
  --target-quantiles 0.01 0.99 \
  --loss-type mae \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --patience 7 \
  --seed 42
