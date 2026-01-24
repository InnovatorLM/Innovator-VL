#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/vlm/xiangan/rice_vl/"}
AIAK_MAGATRON_PATH=${AIAK_MAGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=$1
SAVE=$2

SAVE_LANGUAGE_MODEL=/vlm/xiangan/rice_vl/tmp/language-mcore
SAVE_VISION_MODEL=/vlm/xiangan/rice_vl/tmp/vision-model-mcore
SAVE_ADAPTER=/vlm/xiangan/rice_vl/tmp/adapter-mcore
SAVE_PATCH=/vlm/xiangan/rice_vl/tmp/patch-mcore

TP=$3
PP=$4

# qwen2
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen2_5-vl-32b/qwen2_5.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen2_5-vl-32b/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen2_5-vl-32b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch in vit
python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen2_5-vl-32b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/qwen2_vl/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
