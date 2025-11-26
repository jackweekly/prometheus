# Makefile for Prometheus LLM Project

.PHONY: install test benchmark train-qwen train-bitnet train-spin train-rag clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run syntax checks (basic test)
test:
	python3 -m py_compile training/*.py inference/*.py data/*.py benchmarks/*.py

# Run Benchmarks
benchmark:
	@echo "Running Humanity's Last Exam Benchmark..."
	python3 benchmarks/hle_eval.py
	@echo "Running Vending-Bench..."
	python3 benchmarks/vending_eval.py

# Training Targets
train-qwen:
	python3 training/train.py --mode grpo --config training/config_qwen.yaml

train-bitnet:
	python3 training/train.py --mode grpo --config training/config_bitnet.yaml

train-mamba:
	python3 training/train.py --mode grpo --config training/config_mamba.yaml

train-spin:
	python3 training/train.py --mode spin --config training/config_qwen.yaml

train-rag:
	python3 training/train.py --mode rag --config training/config_qwen.yaml

# Data Generation
generate-data:
	python3 data/generate_data.py

# Clean up (example)
clean:
	rm -rf outputs/*
	rm -rf __pycache__

clean-cache:
	rm -rf ~/.cache/huggingface/hub
