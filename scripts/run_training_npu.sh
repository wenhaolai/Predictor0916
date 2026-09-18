#!/usr/bin/env bash

set -Eeuo pipefail

# Run this script from the repository root:
#   bash Predictor0916/scripts/run_training_npu.sh

# Select the physical Ascend NPU exposed to this training process.
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_CONNECT_TIMEOUT=1800


python Predictor0916/scripts/test_training.py \
  --model-id-or-path /data/models/Qwen3.6-35B-A3B \
  --llm-device npu:0 \
  --llm-device-map balanced \
  --llm-max-memory 0=48GiB 1=48GiB \
  --device npu:0 \
  --torch-dtype float16 \
  --llm-batch-size 1 \
  --data-path Predictor0916/data/qwen-generated.parquet \
  --output-dir Predictor0916/outputs/test_training_npu \
  --validation-ratio 0.2 \
  --num-bins 20 \
  --target-quantiles 0.01 0.99 \
  --loss-type soft_label \
  --lambda-val 0.95 \
  --epochs 10 \
  --batch-size 256 \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --patience 3 \
  --num-workers 4 \
  --pin-memory \
  --seed 42
