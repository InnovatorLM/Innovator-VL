export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH='INNOVATOR_VL_MODEL_PATH'

# General Benchmarks
accelerate launch --num_processes=8 --main_process_port=11191 \
    -m lmms_eval \
    --model innovator_vl \
    --model_args pretrained=$MODEL_PATH,max_pixels=3240000 \
    --tasks ai2d,ai2d_no_mask,ocrbench,chartqa,mmmu_val,mmmu_pro,mmstar,vstar_bench,mmbench_en_dev,mmbench_en_test,mmerealworld,mmerealworld_cn,docvqa_val,infovqa_val,seedbench,seedbench_2_plus,realworldqa \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/innovator_vl_eval/ --verbosity=DEBUG \

# Math & Reasoning Benchmarks
accelerate launch --num_processes=8 --main_process_port=11191 \
    -m lmms_eval \
    --model innovator_vl \
    --model_args pretrained=$MODEL_PATH,max_pixels=3240000,interleave_visuals=True \
    --tasks mathvision_reason_test_reasoning,mathvision_reason_testmini_reasoning,mathverse_testmini_reasoning,mathvista_testmini_cot_reasoning,wemath_testmini_reasoning \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/innovator_vl_eval/ --verbosity=DEBUG \

# Science Benchmarks
accelerate launch --num_processes=8 --main_process_port=11191 \
    -m lmms_eval \
    --model innovator_vl \
    --model_args pretrained=$MODEL_PATH,max_pixels=3240000 \
    --tasks scienceqa,rxnbench_vqa,Molparse,OpenRxn,EMVista,superchem,superchem_cn,SmolInstruct,ProteinLMBench,sfe-en,sfe-zh,microvqa,msearth_mcq,xlrs-lite \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/innovator_vl_eval/ --verbosity=DEBUG \