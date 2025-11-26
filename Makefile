# Makefile for Prometheus LLM Project

.PHONY: install test benchmark run clean clean-cache

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

# Single entrypoint: run the online GRPO loop with Jamba
run:
	python3 training/train.py --config training/config_jamba.yaml

# Data Generation
generate-data:
	python3 data/generate_data.py

# Clean up (example)
clean:
	rm -rf outputs/*
	rm -rf __pycache__

clean-cache:
	rm -rf ~/.cache/huggingface/hub
