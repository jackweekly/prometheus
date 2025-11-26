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
Single entrypoint (online GRPO loop with Jamba):
```bash
make run
```
Tune the HF repo id in `training/config_jamba.yaml` for your chosen Jamba-mini checkpoint.
You can override at runtime with `MODEL_ID=your/model-id make run`. If the repo is gated/private, set `HF_TOKEN` (e.g., `export HF_TOKEN=...` or `huggingface-cli login`).

Ternary (BitNet-style) plan: see `docs/ternary_jamba.md` for experiment ladder (adapters → selective ternary → full ternary), training knobs, and evaluation targets.
Enable ternary hooks by setting `ternary_mode` in `training/config_jamba.yaml` to `attention` (selective) or `full` (attention + SSM).

### Online “learn by doing” loop (no static dataset)
- `training/train.py` now fetches tasks from `fetch_tasks()` (stub) and scores completions with `score_completions()` (exact math or simple code tests). No dataset files are used.
- A human-in-the-loop clarification hook (`ask_human`) is stubbed; replace with your own prompt/feedback mechanism.
- To plug in real web tasks, set `task_url` in `training/config_jamba.yaml`; `fetch_tasks` will HTTP GET JSON `[ { "prompt": "...", "answer": "...", "tests": ["assert ..."] }, ... ]` and fall back to research/local tasks on failure.
- `ask_human` is interactive via stdin; adapt to your UX (e.g., queue/GUI/API) as needed.
- Research mode: set `research_sources` (RSS/Atom feeds like arXiv) and `research_keywords` in `training/config_jamba.yaml`. The loop will create summarize tasks from the feeds and reward summaries that hit the keywords.
- Graceful stop/resume: Ctrl+C saves `state/run_state.json` (iteration, last reward, last tasks). Next run resumes from that iteration.
- Network: HTTP fetches are terminal-based (no browser needed); ensure your cloud shell allows outbound HTTPS.

### Self-improvement harness
- `self_improve.py` applies model-proposed unified diffs safely.
- Defaults to dry-run; use `--apply` to actually patch the working tree.
- Allowed edit targets are limited to project directories/files; adjust `ALLOWED_DIRS` in `self_improve.py` if needed.
Example:
```bash
python self_improve.py --patch-file proposed.diff       # dry run
python self_improve.py --patch-file proposed.diff --apply
```

### Data Generation
To generate a synthetic dataset:
```bash
python data/generate_data.py
```
