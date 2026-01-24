#! /bin/bash
# The script needs to be run on at least 1 nodes.

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-1024}"
MBS="${4:-1}"
GBS="${5:-16}"
NSTEP="${6:-1200}"

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-LLM"}

DATA_PATH=${DATA_PATH:-"/workspace/CogVLM-SFT-311K/CogVLM-SFT-311K/minigpt4/wds/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/vlm/pretrain_models/Qwen2.5-VL-7B-Instruct/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/qwen2_5-vl/qwen2_5-vl-7b-tp${TP}-pp${PP}"}

SAVE_CKPT_PATH="${CHECKPOINT_PATH}/seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps"

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/qwen2_5-vl-7b-tp${TP}-pp${PP}/seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps"}

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# or you can setup qwen2_5-vl-7b by using the following command
MODEL_ARGS=(
    --model-name qwen2_5-vl-7b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer \
    --hf-tokenizer-path $TOKENIZER_PATH \
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules language_model adapter
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 4096
    --init-method-std 0.02
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters $NSTEP
    --lr-decay-iters $NSTEP
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $SAVE_CKPT_PATH
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size ${PP}
    --tensor-model-parallel-size ${TP}
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="/workspace/logs/run_${TM}___tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps"


PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_llm/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${IMG_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee $logfile