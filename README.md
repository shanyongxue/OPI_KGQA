# OPI

This repository provides the implementation of **OPI**, an ontology-guided evidence-path inference framework for multi-hop knowledge graph question answering.

OPI predicts answer-side entity types, retrieves ontology-compatible reasoning paths, and refines final answers through an iterative generator-refiner process.

## Overview

OPI contains three main stages:

1. **Tail-type prediction**: predicts answer-side Freebase-style entity types for each question.
2. **Ontology-guided bidirectional retrieval**: retrieves compact reasoning paths using predicted answer types and ontology relation signatures.
3. **Iterative answer refinement**: uses DeepSeek to revise answers based on retrieved paths, candidate answers, and type constraints.

The full pipeline is provided through two scripts:

```text
scripts/train.sh
scripts/infer.sh
```

The training script builds tail-type supervision data from WebQSP and CWQ, merges them, and trains one unified tail-type prediction model. The inference script runs prediction, retrieval, and refinement on one selected dataset.

## Repository Structure

```text
.
├── config/
│   └── deepspeed_zero3.yml
├── datasets/
├── models/
├── prompts/
├── scripts/
│   ├── train.sh
│   └── infer.sh
├── src/
│   ├── joint_training/
│   │   ├── build_tail_types_dataset_reconstruct.py
│   │   ├── build_finetune_tailtypes.py
│   │   └── joint_finetuning.py
│   ├── qa_prediction/
│   │   ├── gen_tail_types.py
│   │   ├── bidirectional_retrieval.py
│   │   └── iterative_answer_refinement.py
│   └── utils/
├── .gitignore
├── requirements.txt
└── README.md
```

Large files such as datasets, pretrained models, fine-tuned checkpoints, and generated outputs are not tracked by Git. Please download or generate them separately.

## Installation

Create a Python environment:

```bash
conda create -n opienv python=3.10 -y
conda activate opienv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, make sure your PyTorch, CUDA, and DeepSpeed versions are compatible with your hardware.

## Data Preparation

The default pipeline supports **WebQSP** and **CWQ**.

Prepare the following files under `datasets/`:

```text
datasets/
├── RoG-webqsp/data/
├── RoG-cwq/data/
└── ontology_graph_freebase.json
```

The WebQSP and CWQ datasets can be prepared from:

- RoG-WebQSP: https://huggingface.co/datasets/rmanluo/RoG-webqsp
- RoG-CWQ: https://huggingface.co/datasets/rmanluo/RoG-cwq

The ontology graph should be a JSON file of relation signatures:

```json
[
  ["head_type", "relation", "tail_type"]
]
```

## Model Preparation

Prepare the required models under `models/`.

For inference, prepare:

```text
models/
├── OPI_tail_types_llama2/
└── all-mpnet-base-v2/
```

- Fine-tuned OPI tail-type model: https://huggingface.co/Daney/OPI
- Sentence embedding model: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

For training from scratch, additionally prepare:

```text
models/Llama-2-7b-chat-hf/
```

- LLaMA-2-7B-Chat: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## Environment Variables

The iterative answer refinement stage uses a DeepSeek-compatible API.

Export the following variables before running inference:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=your_deepseek_api_url
export OPENAI_MODEL=DeepSeek-V3-250324
```

Do not hard-code API keys in scripts or commit them to GitHub.

## Inference

If you use the released fine-tuned OPI model, you can run inference directly without training from scratch.

Run the full inference pipeline:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=your_api_url
export OPENAI_MODEL=DeepSeek-V3-250324

DATASET=webqsp \
TAIL_MODEL_DIR=models/OPI_tail_types_llama2 \
SBERT_MODEL=models/all-mpnet-base-v2 \
bash scripts/infer.sh
```

Final prediction results are saved under:

```text
outputs/final_predictions/
```

## Training (Optional)

Training is optional if you use the released fine-tuned OPI model.

Run tail-type model training with:

```bash
bash scripts/train.sh
```

The training script performs three steps:

1. builds tail-type supervision data for both WebQSP and CWQ;
2. converts the supervision data into LLaMA-style instruction-tuning data;
3. merges the WebQSP and CWQ training data and fine-tunes one unified tail-type prediction model.

By default, the trained model is saved to:

```text
models/OPI_tail_types_llama2/
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{shan2026opi,
  title  = {Ontology-Guided Evidence-Path Inference for Multi-Hop Knowledge Graph Question Answering},
  author = {Shan, Yongxue and Wu, Meihan and Fang, Cundi and Peng, Jie and Wang, Xiaodong},
  year   = {2026}
}
```