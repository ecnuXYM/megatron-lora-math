#!/usr/bin/env bash
# Launch plain-completion LoRA SFT with explicit framework and data paths.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MILES_DIR:?Set MILES_DIR to your Miles checkout}"
: "${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}"
: "${MODEL_HF:?Set MODEL_HF to a local Qwen2.5-7B model}"
: "${SFT_DATA:?Set SFT_DATA to the generated plain SFT JSONL}"
: "${SAVE_DIR:?Set SAVE_DIR to a new output directory}"
MODEL_CONFIG="$MILES_DIR/scripts/models/qwen2.5-7B.sh"
[[ -f "$MODEL_CONFIG" && -f "$SFT_DATA" && -d "$MODEL_HF" ]] || { echo "Missing model, data or framework config" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT:$MILES_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
source "$MODEL_CONFIG"
ARGS=(
  --hf-checkpoint "$MODEL_HF" --megatron-to-hf-mode bridge
  --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-500}"
  --lora-rank "${LORA_RANK:-8}" --lora-alpha "${LORA_ALPHA:-16}" --lora-dropout 0.0
  --target-modules "${TARGET_MODULES:-q_proj,v_proj}"
  --rollout-function-path own.rollout.plain_sft_rollout.generate_rollout
  --prompt-data "$SFT_DATA" --input-key prompt --metadata-key metadata --rollout-shuffle
  --num-epoch 1 --rollout-batch-size 1 --global-batch-size 1 --loss-type sft_loss
  --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 1024
  --optimizer adam --lr "${LR:-1e-5}" --lr-decay-style cosine --min-lr "${MIN_LR:-1e-6}"
  --lr-warmup-fraction 0.1 --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  --use-distributed-optimizer --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --actor-num-nodes 1 --actor-num-gpus-per-node 2
)
CMD=("${PYTHON:-python3}" "$MILES_DIR/train_async.py" "${MODEL_ARGS[@]}" "${ARGS[@]}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf "%q " "${CMD[@]}"; printf "\n"
else
  cd "$MILES_DIR"
  exec "${CMD[@]}"
fi
