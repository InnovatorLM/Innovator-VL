# sample packing Process
## 1. Data download
Download the required data from [LLaVA-One-Vision-1.5-Mid-Training-85M](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M/tree/main) 
## 2. Configure config.yaml
The following are the key parameter configurations in the config.yaml file:
- **hosts**: Specify the IP for offline packing. For a single machine, only one IP needs to be entered.
- **hf_data**: The address of the downloaded data.
- **directory**: The save address for processing results.
- **checkpoint**: The address of the tokenizer used for VLM.
- **max_len**: The target length for packing.
- **min_pixels** and **max_pixels**: The minimum and maximum pixel values for images input to ViT.
## 3. Execute the Packing Pipeline
Run the offline_packing_pipeline.sh script. This script will sequentially execute the following steps:
1. Parse HuggingFace data.
2. Obtain sample token lengths.
3. Perform packing on the samples.
4. Prepare for generating WebDataset.
5. Package the packing results into WebDataset.
6. Generate the configuration file for WebDataset.






