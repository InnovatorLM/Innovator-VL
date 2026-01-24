# Make adjustments according to the actual case.
OUT_WDS_DIR='/workspace/data4packing/RiceVL/data_procs/emova_captions_wds_300k-8k'
IN_SAMPLE_DIR='/workspace/data4packing/RiceVL/data_procs/raw_packing_data_emova_captions_300k'
PY_EXE="/workspace/AIAK-Training-LLM/tools/data_preprocess/convert_packedsample_to_wds.py"

python -u ${PY_EXE}  --output_dir ${OUT_WDS_DIR}  --json_file ${IN_SAMPLE_DIR}  --video_dir ${IN_SAMPLE_DIR}  --image_dir ${IN_SAMPLE_DIR} --mode caption_pack   --maxcount 2000 2>&1 | tee ./logs/s3_proc_emova_captions_300k-8k.log