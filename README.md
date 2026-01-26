<h1 align="center">
  <img src="asset/logo.png" width="80" align="left">
  Scientific Multimodal Large Language Model for Advanced Reasoning
</h1>

<div align="center">

🤗 **[Models & Datasets](https://huggingface.co/collections/InnovatorLab/innovator-vl)** |
🔗 **[Technical Report](https://arxiv.org/abs/2412.xxxxx)** |
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
  <!-- License -->
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-brightgreen?logo=apache">
  </a>
  <!-- GitHub Stars -->
  <a href="https://github.com/InnovatorLM/Innovator-VL">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/InnovatorLM/Innovator-VL?logo=github">
  </a>
  <!-- Contributors -->
  <a href="https://github.com/InnovatorLM/Innovator-VL/graphs/contributors">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/InnovatorLM/Innovator-VL?logo=github&amp">
  </a>
  <!-- Megatron-LM -->
  <a href="https://github.com/NVIDIA/Megatron-LM">
    <img src="https://img.shields.io/badge/Built%20with-Megatron--LM-76B900?logo=nvidia" alt="Megatron-LM">
  </a>
</p>

---

## 📰 News
...

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
<em><strong>Figure 3:</strong> Performance comparison with state-of-the-art MLLMs on scientific and general benchmarks</em>
</div>

## 🚀 Training Pipeline

### Stage 1: Alignment (Adapter Only)
- **Data**: LLaVA-558K alignment dataset
- **Training**: Vision adapter parameters only
- **Goal**: Align visual and textual representations

### Stage 1.5: Mid-Training (Full Model)
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
# Download instruct dataset (44M samples)
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

### Quick Evaluation

```bash
# Install lmms-eval
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# Run evaluation on scientific benchmarks
accelerate launch --num_processes=8 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=InnovatorLab/Innovator-VL-8B-Instruct,max_pixels=3240000 \
    --tasks=scibench,sci_qa,rxn_bench \
    --batch_size=1
```

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
├── requirements.txt                # Dependencies
├── LICENSE                         # Apache 2.0 License
└── README.md                       # This file
```

## 📝 Citation

If you find Innovator-VL helpful for your research, please consider citing our technical report:

```bibtex

```

## 🙏 Acknowledgments

We express our sincere gratitude to the open-source community for their invaluable contributions that made this work possible:

- **[LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)**: Fully Open Framework for Democratized Multimodal Training
- **[RICE-ViT](https://github.com/deepglint/MVT)**: Advanced visual encoder for fine-grained region understanding
- **[Qwen3](https://github.com/QwenLM/Qwen3)**: Excellent language model backbone
- **[AReaL](https://github.com/inclusionAI/AReaL)**: A Large-Scale Asynchronous Reinforcement Learning Freamwork
- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**: A standardized evaluation framework for Large Multimodal Models
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Efficient, scalable training for large language models
- **[OpenMMReasoner](https://github.com/EvolvingLMMs-Lab/OpenMMReasoner)**: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **[SGLang](https://github.com/sgl-project/sglang)**: SGLang is a fast serving framework for large language models and vision language models
- **[vllm](https://github.com/vllm-project/vllm)**: A high-throughput and memory-efficient inference and serving engine for LLMs

These projects have significantly influenced our work, and we are deeply grateful to their respective authors and contributors.

<div align="center">

**Innovator-VL** - Advancing Scientific Discovery through Multimodal AI

</div>
