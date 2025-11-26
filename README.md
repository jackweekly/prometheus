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
Enable ternary hooks by setting `ternary_mode` in `training/config_jamba.yaml` to `attention` (selective) or `full` (attention + SSM).

### Online “learn by doing” loop (no static dataset)
- `training/train.py` now fetches tasks from `fetch_tasks()` (stub) and scores completions with `score_completions()` (exact math or simple code tests). No dataset files are used.
- A human-in-the-loop clarification hook (`ask_human`) is stubbed; replace with your own prompt/feedback mechanism.
- To plug in real web tasks, replace `fetch_tasks()` with an HTTP fetch in a network-enabled environment.

### Data Generation
To generate a synthetic dataset:
```bash
python data/generate_data.py
```
