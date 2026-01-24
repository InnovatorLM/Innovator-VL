#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ---------------------------
#  环境准备（conda、NCCL等）
# ---------------------------
if [ -f /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh
  conda activate innovator_vl_stable || true
fi

# NCCL / CUDA tuning defaults - 优化NCCL设置
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-mlx5_3}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-50}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}


# ---------------------------
# 默认运行参数（可被外部环境覆盖）
# ---------------------------
export MASTER_ADDR=${MASTER_ADDR:-}
export MASTER_PORT=${MASTER_PORT:-26000}
export WORLD_SIZE=${WORLD_SIZE:-}
export RANK=${RANK:-}
export LOCAL_IP=${LOCAL_IP:-}
export MY_PORT=${MY_PORT:-8469}

# ---------------------------
# 项目路径 / 参数
# ---------------------------
AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/mnt/innovator/code/wenzichen/Innovator-VL}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

TP="${1:-1}" # tensor parallel
PP="${2:-1}" # pipeline parallel
SEQ_LEN="${3:-32768}" # sequence length
MBS="${4:-1}" # micro batch size
GBS="${5:-288}" # global batch size
NSTEP="${6:-171875}" # number of steps, updated for stage 2
SAVE_INTERVAL="${7:-5000}" # save interval

# Stage 2 specific paths - using stage 1.5 checkpoint as starting point
# /mnt/innovator/data/chenshuang/Innovator-VL-Insturct-Data-wds-unpacked/1031  # 25M
DATA_PATH=${DATA_PATH:-"/path/to/dataset/Mix-Stage-1"}  
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/innovator/model/wenzichen/Innovator-VL-8B-stage0"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/checkpoints/stage_1.5_mid_training_innovator_vl_8b"}

# if resume from checkpoint, set the checkpoint path
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-"/mnt/innovator/code/wenzichen/Innovator-VL/checkpoints/0101_SFT_Stage1_44M_Mixed_Sci_Data_V1"}

# set the save checkpoint path and tensorboard path
SAVE_CKPT_PATH=${SAVE_CKPT_PATH:-"/mnt/innovator/code/wenzichen/Innovator-VL/checkpoints/0101_SFT_Stage1_44M_Mixed_Sci_Data_V1"}
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RED='\033[1;31m'
    GREEN='\033[1;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[1;36m'
    NC='\033[0m' # No Color
else
    RED=""
    GREEN=""
    YELLOW=""
    CYAN=""
    NC=""
fi

echo -e "${CYAN}DATA_PATH=${YELLOW}${DATA_PATH}${NC}"
echo -e "${CYAN}TOKENIZER_PATH=${YELLOW}${TOKENIZER_PATH}${NC}"
echo -e "${CYAN}CHECKPOINT_PATH=${YELLOW}${CHECKPOINT_PATH}${NC}"
echo -e "${CYAN}RESUME_FROM_CHECKPOINT=${YELLOW}${RESUME_FROM_CHECKPOINT}${NC}"
echo -e "${CYAN}SAVE_CKPT_PATH=${YELLOW}${SAVE_CKPT_PATH}${NC}"
echo -e "${CYAN}TENSORBOARD_PATH=${YELLOW}${TENSORBOARD_PATH}${NC}"

# ---------------------------
# 可选静态 IP 列表
# ---------------------------
declare -a list_ip=(
    # "172.27.50.128"
    # "172.27.94.195"
)

# ---------------------------
# 帮助函数
# ---------------------------
first_non_empty() {
  for v in "$@"; do
    if [[ -n "${v:-}" ]]; then
      echo "$v"
      return 0
    fi
  done
  return 1
}

detect_current_ip() {
  if [[ -n "${ATLAS_NODE_IP:-}" ]]; then
    echo "${ATLAS_NODE_IP}"
    return
  fi
  if [[ -n "${HOST_IP:-}" ]]; then
    echo "${HOST_IP}"
    return
  fi
  if [[ -n "${PREFERRED_IF:-}" ]]; then
    ip -4 addr show "$PREFERRED_IF" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1 && return
  fi
  for ifc in eth0 ib0 ens3 ens4 ens5 mlx5_0 mlx5_1; do
    ip -4 addr show "$ifc" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1 && return
  done
  hostname -I 2>/dev/null | awk '{print $1}' && return
  hostname -i 2>/dev/null | awk '{print $1}' || true
}

# ---------------------------
# 自动识别分布式信息
# ---------------------------
echo "=== Distributed env auto-detection start ==="
CURRENT_IP=${LOCAL_IP:-$(detect_current_ip)}
echo "Detected CURRENT_IP: ${CURRENT_IP:-<none>}"

if [[ -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" && -n "${RANK:-}" ]]; then
  echo "Using pre-set MASTER_ADDR/WORLD_SIZE/RANK from environment."
  echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}, WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"
else
  if [[ -n "${SLURM_NODELIST:-}" ]]; then
    if command -v scontrol >/dev/null 2>&1; then
      MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
    fi
    WORLD_SIZE=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
    RANK=${SLURM_NODEID:-0}
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    CURRENT_IP=${ATLAS_NODE_IP:-${LOCAL_IP:-$CURRENT_IP}}
    echo "Detected SLURM environment. MASTER_ADDR=${MASTER_ADDR}, WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"
  fi
fi

if [[ -z "${WORLD_SIZE:-}" || -z "${RANK:-}" ]]; then
  if [[ ${#list_ip[@]} -gt 0 ]]; then
    NNODES=${#list_ip[@]}
    WORLD_SIZE=${WORLD_SIZE:-$NNODES}
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
      if [[ "${list_ip[$i]}" == "${CURRENT_IP}" || "${list_ip[$i]}" == "${MASTER_ADDR:-}" ]]; then
        NODE_RANK=$i
        break
      fi
    done
    if [[ $NODE_RANK -ne -1 ]]; then
      RANK=${RANK:-$NODE_RANK}
      echo "Matched CURRENT_IP to static list_ip index ${NODE_RANK} -> RANK=${RANK}"
    else
      echo "Warning: Current IP (${CURRENT_IP}) not found in static list_ip. Will try to continue."
    fi
  fi
fi

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-${list_ip[0]:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-26000}

echo "Final distributed config:"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  WORLD_SIZE=${WORLD_SIZE}"
echo "  RANK=${RANK}"
echo "  CURRENT_IP=${CURRENT_IP}"

MASTER_IPV4=$(ping -c 1 "$MASTER_ADDR" 2>/dev/null | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -n1 || true)
MASTER_IPV4=${MASTER_IPV4:-$MASTER_ADDR}
echo "Resolved MASTER_IPV4: ${MASTER_IPV4}"
export HEAD_NODE_ADDRESS=$MASTER_IPV4
export MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_IP

# ---------------------------
# Proxy / NO_PROXY 设置
# ---------------------------
export http_proxy=${http_proxy:-http://10.20.112.35:3143}
export https_proxy=${https_proxy:-http://10.20.112.35:3143}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export NO_PROXY=${NO_PROXY:-"127.0.0.1,localhost,::1,${HEAD_NODE_ADDRESS}"}
export no_proxy=${no_proxy:-$NO_PROXY}
echo "NO_PROXY set to: $NO_PROXY"

# ---------------------------
# 日志目录和 checkpoint
# ---------------------------
mkdir -p "$SAVE_CKPT_PATH" "$TENSORBOARD_PATH" "$SAVE_CKPT_PATH/dataloader"
chmod 777 "$TENSORBOARD_PATH"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
  DISTRIBUTED_ARGS=( --nproc_per_node "$GPUS_PER_NODE" )
else
  DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$WORLD_SIZE"
    --node_rank "$RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
  )
fi


# ---------------------------
# 训练参数
# ---------------------------
MODEL_ARGS=( --model-name innovator-vl-8b )

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
    # --packing-pretrain-data
    # --packing-batch-size 10000
)

# Stage 2 specific training args
TRAINING_ARGS=(
    --image-resolution 1000
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --enable-discard-sample
    # --record-memory-history
    # --memory-snapshot-path "/mnt/innovator/code/wenzichen/Innovator-VL/checkpoints/memory_snapshot.pickle"
)

LOAD_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [ -d "$RESUME_FROM_CHECKPOINT" ] && [ -f "$RESUME_FROM_CHECKPOINT/latest_checkpointed_iteration.txt" ]; then
        echo "Resuming training from checkpoint: $RESUME_FROM_CHECKPOINT"
        echo "WARNING: When resuming, ensure the following parameters match the original training:"
        echo "  - TP (tensor parallel): ${TP}"
        echo "  - PP (pipeline parallel): ${PP}"
        echo "  - MBS (micro batch size): ${MBS}"
        echo "  - GBS (global batch size): ${GBS}"
        echo "  - WORLD_SIZE: ${WORLD_SIZE}"
        echo "  - GPUS_PER_NODE: ${GPUS_PER_NODE}"
        LOAD_ARGS=(--load "$RESUME_FROM_CHECKPOINT")
    else
        echo "Warning: Checkpoint path specified in RESUME_FROM_CHECKPOINT is not valid."
        echo "It must be a directory containing a 'latest_checkpointed_iteration.txt' file."
        echo "Provided path: $RESUME_FROM_CHECKPOINT"
        echo "Falling back to initial model: $CHECKPOINT_PATH"
    fi
fi

if [ ${#LOAD_ARGS[@]} -eq 0 ]; then
    echo "Starting training from initial model: $CHECKPOINT_PATH"
    LOAD_ARGS=(--load "$CHECKPOINT_PATH")
fi

TRAINING_ARGS+=(
    "${LOAD_ARGS[@]}"
    --save "$SAVE_CKPT_PATH"
    --save-interval $SAVE_INTERVAL
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --override-opt_param-scheduler
    # --eval-interval 50        
    # --eval-iters 10          
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    # --tensorboard-log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT:-default}"
        --wandb-exp-name "${WANDB_NAME:-run}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

# Stage 2 specific environment variables
export OFFLINE_PACKED_DATA='0'
export OFFLINE_PACKING_VQA='0'
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.72}

# network tweaks
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

# ---------------------------
# Launch
# ---------------------------
echo "Launching Stage 2 Instruction Tuning with torchrun:"
echo "torchrun ${DISTRIBUTED_ARGS[*]} $AIAK_TRAINING_PATH/aiak_training_llm/train.py ..."

PYTHONPATH="${PYTHONPATH:-}"
PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"


