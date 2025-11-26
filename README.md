# Efficient Self-Improving LLM Foundation (Jamba-only)

This project provides the foundation for training and deploying efficient, self-improving Large Language Models (LLMs) using Unsloth (GRPO) and Distilabel.

## Structure

- `training/`: Configuration and scripts for training Jamba.
- `data/`: Scripts for generating synthetic datasets.
- `inference/`: Scripts for running inference.
- `docs/ternary_jamba.md`: Playbook for BitNet-style ternary experiments on Jamba-mini.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
GRPO LoRA fine-tune:
```bash
python training/train.py --config training/config_jamba.yaml
```
Tune the HF repo id in `training/config_jamba.yaml` for your chosen Jamba-mini checkpoint.

Ternary (BitNet-style) plan: see `docs/ternary_jamba.md` for experiment ladder (adapters → selective ternary → full ternary), training knobs, and evaluation targets.

### Data Generation
To generate a synthetic dataset:
```bash
python data/generate_data.py
```
