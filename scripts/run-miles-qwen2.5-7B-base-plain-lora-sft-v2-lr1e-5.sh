#!/usr/bin/env bash
set -euo pipefail

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MILES_DIR=${MILES_DIR:-/root/miles}
MODEL_HF=${MODEL_HF:-/root/models/Qwen2.5-7B}
SFT_DATA=${SFT_DATA:-/root/own/data/math_plain_sft_l3_4_core_train2000_seed5151.jsonl}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_RANK * 2))}
TARGET_MODULES=${TARGET_MODULES:-q_proj,v_proj}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
RUN_TAG=${RUN_TAG:-r${LORA_RANK}_qv_lr1e-5}
SAVE_DIR=${SAVE_DIR:-/root/models/Qwen2.5-7B_lora_sft_base_plain_v2_${RUN_TAG}}

cd "${MILES_DIR}"
source scripts/models/qwen2.5-7B.sh

ray stop --force || true
pkill -9 ray || true

ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root:/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}'

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_HF}"
  --megatron-to-hf-mode bridge
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

LORA_ARGS=(
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout 0.0
  --target-modules "${TARGET_MODULES}"
)

SFT_ARGS=(
  --rollout-function-path own.rollout.plain_sft_rollout.generate_rollout
  --prompt-data "${SFT_DATA}"
  --input-key prompt
  --metadata-key metadata
  --rollout-shuffle
  --num-epoch 1
  --rollout-batch-size 1
  --global-batch-size 1
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 1024
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction 0.1
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
  --use-distributed-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
