# Efficient Self-Improving LLM Foundation

This project provides the foundation for training and deploying efficient, self-improving Large Language Models (LLMs) using Unsloth (GRPO) and Distilabel.

## Structure

- `training/`: Configuration and scripts for training models (Qwen, BitNet).
- `data/`: Scripts for generating synthetic datasets.
- `inference/`: Scripts for running inference.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train a model using GRPO:
```bash
python training/train_grpo.py --config training/config_qwen.yaml
```

### Data Generation
To generate a synthetic dataset:
```bash
python data/generate_data.py
```
