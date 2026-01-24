### CAPTIONS Workflow
- Step-1: `python s1_get_tokenlens_v2.py --config ./configs/s1_config_emova_captions_300k-8k.yaml`
- Step-2: `sandbox_packing_captions.ipynb`
    - section `Step2-1`: bin-packing processing (You can set different bin-packing strategies and their parameters based on your understanding of the element distribution in the hash_buckets.).
    - section `hash_buckets‘s element distribution`: Visualize the element distribution in hash_buckets to help design bin-packing strategies.
    - section `Step2-2`: Convert pre-packing format to post-packing format.
- Step-3: `bash s3_test-emova_captions_300k-8k.sh`
- Step-4: `export OFFLINE_PACKED_DATA='1'`(Settings in the training script remain unchanged).