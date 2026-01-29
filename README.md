<h1 align="center">
  <img src="asset/logo.png" width="80" align="left">
  Scientific Multimodal Large Language Model for Advanced Reasoning
</h1>

<div align="center">

🤗 **[Models & Datasets](https://huggingface.co/collections/InnovatorLab/innovator-vl)** |
🔗 **[Technical Report](https://arxiv.org/pdf/2601.19325)** |
🖥️ **[Demo](https://huggingface.co/spaces/InnovatorLab/Innovator-VL)**

</div>

<p align="center">
  <!-- Instruct Model Downloads -->
  <a href="https://huggingface.co/InnovatorLab/Innovator-VL-8B-Instruct">
    <img alt="HF Model Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/InnovatorLab/Innovator-VL-8B-Instruct&amp;query=downloads&amp;label=Innovator-VL-8B-Instruct%20Downloads&amp;color=yellow&amp;logo=huggingface&amp">
  </a>
  <!-- Thinking Model Downloads -->
  <a href="https://huggingface.co/InnovatorLab/Innovator-VL-8B-Thinking">
    <img alt="HF Model Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/InnovatorLab/Innovator-VL-8B-Thinking&amp;query=downloads&amp;label=Innovator-VL-8B-Thinking%20Downloads&amp;color=yellow&amp;logo=huggingface&amp">
  </a>
  <!-- Instruct Dataset Downloads -->
  <a href="https://huggingface.co/datasets/InnovatorLab/Innovator-VL-Instruct-46M">
    <img alt="HF Instruct Dataset Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/InnovatorLab/Innovator-VL-Instruct-46M&amp;query=downloads&amp;label=Instruct%20DATA%20Downloads&amp;color=blue&amp;logo=huggingface&amp">
  </a>
  <!-- RL Dataset Downloads -->
  <a href="https://huggingface.co/datasets/InnovatorLab/Innovator-VL-RL-172K">
    <img alt="RL Data Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/InnovatorLab/Innovator-VL-RL-172K&amp;query=downloads&amp;label=RL%20DATA%20Downloads&amp;color=blue&amp;logo=huggingface&amp">
  </a>
  <!-- Collection -->
  <a href="https://huggingface.co/collections/InnovatorLab/innovator-vl" target="_blank">
  <img alt="Hugging Face Collection" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-ffd21e">
  </a>
  <!-- License -->
  <a href="LICENSE">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <!-- GitHub Stars -->
  <a href="https://github.com/InnovatorLM/Innovator-VL">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/InnovatorLM/Innovator-VL?logo=github">
  </a>
  <!-- Contributors -->
  <a href="https://github.com/InnovatorLM/Innovator-VL/graphs/contributors">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/InnovatorLM/Innovator-VL?logo=github&amp">
  </a>
  <!-- PRs Welcome -->
  <a href="https://github.com/InnovatorLM/Innovator-VL/pulls">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github&amp">
  </a>
  <!-- Megatron-LM -->
  <a href="https://github.com/NVIDIA/Megatron-LM">
    <img src="https://img.shields.io/badge/Built%20with-Megatron--LM-76B900?logo=nvidia" alt="Megatron-LM">
  </a>
</p>

---

## 📰 News
* **`2026.1.29`** 🤗🤗 Our model has been officially released at the ["Agentic Science at Scale" workshop](https://mp.weixin.qq.com/s/8yl4bK8DL2UIsreXcjdbvg)!
* **`2026.1.29`** 🤗🤗 We’ve already adapted it to [SciMaster](https://scimaster.bohrium.com/). Come and give it a try!
* **`2026.1.28`** 🤗🤗 We release the [Innovator-VL Technical Report](https://arxiv.org/pdf/2601.19325)!
* **`2026.1.26`** 🤗🤗 We release all of our code, model, data, and pipeline! Check out the [InnovatorVL collection on Hugging Face](https://huggingface.co/collections/InnovatorLab/innovator-vl)


## 📖 Table of Contents
- [Introduction](#introduction)
- [Models & Checkpoints](#models--checkpoints)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Performance](#performance)
- [Training Pipeline](#training-pipeline)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Directory Structure](#directory-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Introduction

**Innovator-VL** is a scientific multimodal large language model designed to advance multimodal understanding and reasoning across diverse scientific domains. Contrary to conventional approaches that rely on massive scientific domain-specific pretraining, Innovator-VL demonstrates **remarkable data efficiency**, achieving competitive performance using fewer than **five million** carefully curated scientific samples. Meanwhile, we provide a fully transparent and end-to-end reproducible training pipeline for scientific multimodal modeling, covering all stages from data collection and cleaning to preprocessing, supervised fine-tuning, reinforcement learning, and evaluation. 

### Key Highlights

🚀 **Superior Performance**: State-of-the-art results on scientific benchmarks with 8B parameter model

💡 **Data Efficiency**: Competitive performance without large-scale scientific pretraining

🔬 **Strong Generalization**: Maintains excellent performance on general vision and reasoning tasks

📊 **Fully Transparent**: Reproducible training methodology from data collection to evaluation


## 📦 Models & Checkpoints

| Model | Base LLM | Size | Link | Training Log |
|-------|----------|------|------|--------------|
| **Innovator-VL-8B-Instruct** | Qwen3-8B | 8B | [🤗 HF](https://huggingface.co/InnovatorLab/Innovator-VL-8B-Instruct) | Available |
| **Innovator-VL-8B-Thinking** | Qwen3-8B | 8B | [🤗 HF](https://huggingface.co/InnovatorLab/Innovator-VL-8B-Thinking) | Available |

## 📊 Datasets

### Training Datasets

<div align="center">
<img src="asset/dataset.png" width="100%" alt="Training Dataset">
<br>
<em><strong>Figure 1:</strong> Data distribution across different training stages</em>
</div>

<br>

| Dataset | Size | Description | Status |
|---------|------|-------------|--------|
| **LLaVA-558K** | 558K | Alignment dataset for Stage 1 | [🤗 HF](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| **LLaVA-OneVision-1.5-Mid-Training-85M** | 85M | Diverse multimodal samples for Stage 1.5 | [🤗 HF](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-1.5-Mid-Training-85M) |
| **Innovator-VL-Instruct-46M** | 46M | Instruction-following samples for Stage 2 | [🤗 HF](https://huggingface.co/datasets/InnovatorLab/Innovator-VL-Instruct-46M) |
| **Innovator-VL-RL-172K** | 172K | Discrepancy-driven RL dataset | [🤗 HF](https://huggingface.co/datasets/InnovatorLab/Innovator-VL-RL-172K) |

## 🏗️ Architecture

<div align="center">
<img src="asset/architecture_01.png" width="100%" alt="Innovator-VL Architecture">
<br>
<em><strong>Figure 2:</strong> Innovator-VL architecture with RICE-ViT visual encoder, PatchMerger, and Qwen3 language decoder</em>
</div>

<br>

Innovator-VL adopts a principled architecture design optimized for scientific understanding:

**Visual Encoder**: RICE-ViT captures fine-grained, region-level semantics for accurate perception of structured visual elements (symbols, annotations, relational components)

**Vision-Language Projector**: PatchMerger balances representational capacity and computational efficiency by merging visual patches into compact yet semantically informative representations

**Language Decoder**: Qwen3-8B-Base provides a strong foundation for reasoning and generation, pre-trained on a broad and diverse corpus



## 📊 Performance

<div align="center">
<img src="asset/bar_figure_01.png" width="100%" alt="Performance Comparison">
<br>
<em><strong>Figure 3:</strong> Performance comparison with state-of-the-art MLLMs on general vision, reasoning, anb scientific benchmarks</em>
</div>

<br>

<div align="center">
<img src="asset/main_results.png" width="100%" alt="main results">
<br>
<em><strong>Figure 4:</strong> Detailed comparison with state-of-the-art MLLMs across general, mathematical reasoning, and scientific benchmarks</em>
</div>

## 🚀 Training Pipeline

### Stage 1: Alignment
- **Data**: LLaVA-558K alignment dataset
- **Training**: Vision adapter parameters only
- **Goal**: Align visual and textual representations

### Stage 1.5: Mid-Training
- **Data**: 85M high-quality multimodal samples
- **Training**: Full model
- **Goal**: Enhance multimodal understanding

### Stage 2: Instruct Tuning
- **Data**: 46M instruction data
- **Training**: Full model
- **Goal**: Develop instruction-following capabilities

### RL: GSPO Optimization
- **Algorithm**: Group Sequence Policy Optimization
- **Data**: 172K discrepancy-driven RL samples
- **Goal**: Enhancing model reasoning and the ability to solve complex scientific problems


## Inference with HuggingFace
```python
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
model_path = "InnovatorLab/Innovator-VL-8B-Instruct"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```


## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/InnovatorLab/Innovator-VL.git
cd Innovator-VL

# Install dependencies
pip install -r requirements.txt
```

### Model Conversion

```bash
# HF → Megatron
bash examples/innovator_vl/convert/convert_8b_hf_to_mcore.sh

# Megatron → HF
bash examples/innovator_vl/convert/convert_8b_mcore_to_hf.sh
```

### Stage 1 Training

```bash
# Download LLaVA-558K dataset
# Set environment variables and run
AIAK_TRAINING_PATH=/path/to/Innovator-VL \
DATA_PATH=/path/to/LLaVA-558K \
TOKENIZER_PATH=/path/to/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/path/to/Innovator-VL-8B-stage0 \
bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh
```

### Stage 1.5 Training

```bash
# Download Mid-Training dataset (85M samples)
AIAK_TRAINING_PATH=/path/to/Innovator-VL \
DATA_PATH=/path/to/LLaVA-OneVision-1.5-Mid-Training-85M \
TOKENIZER_PATH=/path/to/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/path/to/stage_1_output \
bash examples/innovator_vl/quick_start/train_auto_stage_1.5_mid_training_innovator_vl_8b.sh
```

### Stage 2 Training

```bash
# Download instruct dataset (46M samples)
AIAK_TRAINING_PATH=/path/to/Innovator-VL \
DATA_PATH=/path/to/Innovator-VL-Instruct-46M \
TOKENIZER_PATH=/path/to/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/path/to/stage_1.5_output \
bash examples/innovator_vl/quick_start/train_auto_stage_2_instruct_innovator_vl_8b.sh
```

### RL Training (GSPO)

```bash
cd RL/train_scripts
# Configure paths in configs/innovator-vl-8b-gspo.yaml
bash run_example.sh
```

## 🔬 Evaluation

Innovator-VL is evaluated using the **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** framework, enabling standardized, reproducible evaluation across **general vision**, **mathematical reasoning**, and **scientific domains**.

We provide an official evaluation script that covers all supported benchmarks.

### 📜 Evaluation Script

An official evaluation script is provided:

👉 **[`lmms-eval/eval_innovator_vl.sh`](lmms-eval/eval_innovator_vl.sh)**

Before running the evaluation, please set the model path:

```bash
export MODEL_PATH=/path/to/Innovator-VL-8B-Instruct (or Innovator-VL-8B-Thinking)
```

Then launch the evaluation:

```
cd lmms-eval
bash eval_innovator_vl.sh
```

### 🧪 Supported Benchmarks

The evaluation script is organized into three major categories.

#### 🌍 General Vision Benchmarks

These benchmarks evaluate perception, OCR, document understanding, and real-world multimodal reasoning:

- **AI2D**: `ai2d`, `ai2d_no_mask`
- **OCRBench**
- **ChartQA**
- **MMMU**: `mmmu_val`, `mmmu_pro_standard`
- **MMStar**
- **VStar-Bench**
- **MMBench**: `mmbench_en_dev`, `mmbench_en_test`
- **MME-RealWorld**: `mmerealworld`, `mmerealworld_cn`
- **DocVQA**: `docvqa_val`
- **InfoVQA**: `infovqa_val`
- **SEEDBench**: `seedbench`, `seedbench_2_plus`
- **RealWorldQA**

#### ➗ Math & Reasoning Benchmarks

These benchmarks focus on mathematical reasoning and chain-of-thought multimodal problem solving:

- **MathVision**: `mathvision_reason_test_reasoning`, `mathvision_reason_testmini_reasoning`
- **MathVerse**: `mathverse_testmini_reasoning`
- **MathVista**: `mathvista_testmini_cot_reasoning`
- **WeMath**: `wemath_testmini_reasoning`

#### 🔬 Scientific Benchmarks

These benchmarks evaluate Innovator-VL’s capabilities across chemistry, biology, medicine, earth science, and scientific document understanding:

- **General Science & Scientific Cognition**: `sfe-en`, `sfe-zh`, `scienceqa`
- **Chemistry & Molecular Science**: `rxnbench_vqa`, `OpenRxn`, `MolParse`, `SmolInstruct`, `superchem`, `superchem_cn`
- **Biology, Medicine & Microscopy**: `EMVista`, `microvqa`, `ProteinLMBench`
- **Earth Science & Remote Sensing**: `msearth_mcq`, `xlrs-lite`

## 📁 Directory Structure

```
Innovator-VL/
├── examples/innovator_vl/          # Training examples and scripts
│   ├── quick_start/                # Training scripts for all stages
│   ├── convert/                    # Model conversion utilities
│   └── evaluate/                   # Evaluation scripts
│
├── aiak_training_llm/              # Core training framework
│   ├── models/innovator_vl/        # Model implementations
│   │   ├── innovator_vl_model.py
│   │   ├── innovator_vl_config.py
│   │   ├── adapter.py
│   │   └── vision_model.py
│   └── train.py                    # Main training script
│
├── tools/                          # Utilities
│   └── convert_checkpoint/         # Checkpoint conversion tools
│       └── custom/innovator_vl/    # Model-specific converters
│
├── RL/                             # RL training framework
│   ├── train_scripts/              # RL training scripts
│   ├── configs/                    # RL configurations (GSPO)
│   ├── trains/                     # RL training core
│   ├── engine/                     # RL training engine
│   ├── reward/                     # Reward functions
│   └── 3rdparty/                   # Third-party dependencies
│
├── ds/                             # Custom training implementations
│   └── innovator_vl/               # Model definitions
│       ├── configuration_innovator_vl.py
│       └── modeling_innovator_vl.py
│
├── asset/                          # Assets (logos, figures)
│   ├── logo.png
│   ├── architecture.pdf
│   ├── performance.pdf
│   └── data_pipeline.pdf
│
├── requirements.txt                
├── LICENSE                         
└── README.md                      
```

## 📝 Citation

If you find Innovator-VL helpful for your research, please consider citing our technical report:

```bibtex
@article{wen2026innovator,
  title={Innovator-VL: A Multimodal Large Language Model for Scientific Discovery},
  author={Wen, Zichen and Yang, Boxue and Chen, Shuang and Zhang, Yaojie and Han, Yuhang and Ke, Junlong and Wang, Cong and others},
  journal={arXiv preprint arXiv:2601.19325},
  year={2026}
}
```

## 🙏 Acknowledgments

We express our sincere gratitude to the open-source community for their invaluable contributions that made this work possible:

- **[LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)**: Fully Open Framework for Democratized Multimodal Training
- **[RICE-ViT](https://github.com/deepglint/MVT)**: Advanced visual encoder for fine-grained region understanding
- **[Qwen3](https://github.com/QwenLM/Qwen3)**: Excellent language model backbone
- **[AReaL](https://github.com/inclusionAI/AReaL)**: A Large-Scale Asynchronous Reinforcement Learning Freamwork
- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**: A standardized evaluation framework for Large Multimodal Models
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Efficient, scalable training for large language models
- **[Bee](https://huggingface.co/datasets/Open-Bee/Honey-Data-15M)**: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs
- **[OpenMMReasoner](https://github.com/EvolvingLMMs-Lab/OpenMMReasoner)**: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **[SGLang](https://github.com/sgl-project/sglang)**: SGLang is a fast serving framework for large language models and vision language models
- **[vLLM](https://github.com/vllm-project/vllm)**: A high-throughput and memory-efficient inference and serving engine for LLMs

These projects have significantly influenced our work, and we are deeply grateful to their respective authors and contributors.

<div align="center">

**Innovator-VL** - Advancing Scientific Discovery through Multimodal AI

</div>
