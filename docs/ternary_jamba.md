# Ternary Jamba Strategy (BitNet-style on Jamba-mini)

This document turns the BitNet b1.58 ideas into a concrete Jamba path. Jamba is hybrid (attention + SSM), so we stage experiments from safest to riskiest.

## Experiment ladder

1) **Ternary adapters only (safe)**
   - Freeze base Jamba-mini.
   - Train LoRA adapters; export them as ternary (round/clip to {-1,0,1} with absmean scaling).
   - Pros: low risk, adapter matmuls become add/sub; keeps base quality.
   - How: train normally, then ternarize adapter weights offline; or add a post-training hook that replaces adapter weights with ternary values.

2) **Selective ternary (balanced)**
   - Ternarize attention projections + MLP (`q/k/v/o`, `gate/up/down`).
   - Keep SSM projections (`in_proj`, `x_proj`, `dt_proj`) at 4–8 bits to avoid destabilizing the scan.
   - Inject learnable RMSNorm before each ternarized linear; fine-tune with STE.

3) **Full ternary (high risk/reward)**
   - RMSNorm injection before every linear (attention + SSM).
   - STE + absmean scaling per weight matrix.
   - Start with shorter contexts (4k–8k) during “healing,” then ramp to 16k+.

## Training knobs

- **Quantizer**: `w_q = clamp(round(w / (absmean(w)+eps)), -1, 1)`; forward uses `w_q`; backward uses STE (`w + (w_q - w).detach()`).
- **Optimizer**: AdamW/Adafactor; LR 3e-5–1e-4; warmup 5–10% steps; grad clip 0.5–1.0.
- **Regularizers**: weight decay on latent weights; optional sparsity penalty to encourage zeros.
- **Curriculum**: shorter seq first; expand after loss stabilizes. Consider two-stage (attention-only ternary → add SSM).

## Evaluation checklist

- Quality: perplexity on held-out; reasoning (GSM8K-lite); long-context recall (LongBench subset).
- Efficiency: tokens/s vs 4-bit baseline; VRAM/CPU RAM; power proxy (nvidia-smi or psutil on M-series); stability on 100k+ contexts.
- Acceptance: do not regress reasoning >X% vs 4-bit baseline; achieve measurable t/s or power gains.

## Practical notes

- Kernels: no bitnet.cpp for Jamba; ternary SSM needs custom kernels (CPU easier, GPU needs custom scan). Prototype in PyTorch first to verify quality.
- Export: GGUF/GGML don’t support ternary here; plan a bespoke runner if you go beyond adapters.
- Data: reuse your GRPO pipeline; for healing, keep group size small (4) and LR conservative.
