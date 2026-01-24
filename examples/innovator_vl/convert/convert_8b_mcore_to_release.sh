set -x

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/mnt/innovator/code/wenzichen/Innovator-VL}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

cd /mnt/innovator/code/wenzichen/Innovator-VL/

if [ -f /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh
  conda activate innovator_vl_stable || true
fi

LOAD=$1
SAVE=$2
# TP=$3
# PP=$4


bash $AIAK_TRAINING_PATH/examples/innovator_vl/convert/convert_8b_mcore_to_hf.sh \
    $LOAD tmp_hf 2 1

bash $AIAK_TRAINING_PATH/examples/innovator_vl/convert/convert_8b_hf_to_mcore.sh \
    tmp_hf $SAVE 4 1

rm -rf tmp_hf
