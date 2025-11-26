import argparse
import yaml
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Ternary helpers ---
def ternary_weight(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Absmean scaling + STE ternary projection to {-1, 0, 1}."""
    scale = w.detach().abs().mean()
    w_hat = torch.clamp(torch.round(w / (scale + eps)), -1, 1)
    # STE: gradient is 1 through w
    return w + (w_hat - w).detach()


def attach_ternary_forward(module: torch.nn.Module):
    """Replace Linear forward with ternary-weighted linear via a forward hook."""
    def hook(_module, inputs, output=None):
        x = inputs[0]
        w_q = ternary_weight(_module.weight)
        return torch.nn.functional.linear(x, w_q, _module.bias)
    module.register_forward_hook(hook)


def apply_ternary_hooks(model, mode: str, attention_keys, ssm_keys):
    """
    mode: "none" | "attention" | "full"
    attention_keys: substrings to match attention/MLP projections
    ssm_keys: substrings to match SSM projections
    """
    if mode == "none":
        print("Ternary mode: none (standard training).")
        return

    def should_hook(name: str):
        if mode == "full":
            return any(k in name for k in attention_keys + ssm_keys)
        if mode == "attention":
            return any(k in name for k in attention_keys)
        return False

    hooked = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and should_hook(name):
            attach_ternary_forward(module)
            hooked += 1
    print(f"Ternary mode '{mode}': attached hooks to {hooked} Linear modules.")


def ask_human(queries: List[str]) -> List[str]:
    """Placeholder for human-in-the-loop clarification. Currently just logs and returns empty answers."""
    print("Human clarification requested for queries:")
    for q in queries:
        print(f"- {q}")
    return [""] * len(queries)


def fetch_tasks() -> List[Dict[str, Any]]:
    """
    Task source stub. In a network-enabled environment, replace this with HTTP fetch.
    Returns a list of dicts with keys: prompt, answer (for auto-eval), and optional tests.
    """
    # Simple math/code-style tasks for self-play
    return [
        {"prompt": "Compute 17 * 23.", "answer": "391"},
        {"prompt": "Write a Python function add(a, b) that returns their sum.", "answer": "def add(a, b):\n    return a + b", "tests": ["assert add(1,2)==3", "assert add(-1,1)==0"]},
    ]


def score_completions(tasks: List[Dict[str, Any]], completions: List[str]) -> List[float]:
    """
    Simple reward: exact string contains for math; code tested via exec if tests provided.
    If low confidence, asks human (stub) for guidance.
    """
    rewards = []
    ask = []
    ask_idx = []
    for i, (task, comp) in enumerate(zip(tasks, completions)):
        if "tests" in task:
            local_vars = {}
            try:
                exec(comp, {}, local_vars)  # Unsafe in general; acceptable for controlled tests
                passed = True
                for t in task["tests"]:
                    exec(t, {}, local_vars)
                rewards.append(1.0 if passed else 0.0)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0 if str(task["answer"]) in comp else 0.0)
        if rewards[-1] == 0.0:
            ask.append(f"Clarify/guide for task: {task['prompt']}")
            ask_idx.append(i)
    if ask:
        human_answers = ask_human(ask)
        for idx, hint in zip(ask_idx, human_answers):
            # If human gives a hint, provide small reward bump to encourage exploration
            if hint.strip():
                rewards[idx] += 0.2
    return rewards


# --- GRPO Logic ---
def train_grpo(model, tokenizer, config):
    print("Starting GRPO Training (online tasks, no static dataset)...")

    tasks = fetch_tasks()

    # Generate completions for each task
    prompts = [t["prompt"] for t in tasks]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    rewards = score_completions(tasks, completions)

    training_args = GRPOConfig(
        output_dir = config['output_dir'],
        learning_rate = float(config['learning_rate']),
        per_device_train_batch_size = config['batch_size'],
        gradient_accumulation_steps = config['gradient_accumulation_steps'],
        max_steps = config['max_steps'],
        warmup_steps = config['warmup_steps'],
        logging_steps = config['logging_steps'],
        save_steps = config['save_steps'],
        max_prompt_length = 512,
        max_completion_length = 1024,
        num_generations = config.get('group_size', 4),
        report_to = "wandb",
        dataloader_num_workers = config.get('dataloader_num_workers', 4), # Use system RAM/CPU for data loading
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [lambda p, c, **kwargs: rewards],
        args = training_args,
        train_dataset = None, # Online tasks; no persistent dataset
    )
    # trainer.train()
    print("GRPO Training setup complete (Placeholder execution with online tasks).")

def main():
    parser = argparse.ArgumentParser(description="Jamba-only GRPO Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loading model: {config['model_name']} (Jamba-only pipeline)")

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print("Detected Jamba model. Loading with HF + PEFT for hybrid attention/SSM layers.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    target_modules = config.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "x_proj",
            "dt_proj",
        ],
    )

    ternary_mode = config.get("ternary_mode", "none")  # "none" | "attention" | "full"
    apply_ternary_hooks(
        model,
        mode=ternary_mode,
        attention_keys=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ssm_keys=["in_proj", "x_proj", "dt_proj"],
    )

    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    print("Gradient checkpointing enabled.")
    
    # Torch Compile Optimization
    # print("Compiling model with torch.compile...")
    # model = torch.compile(model)
    train_grpo(model, tokenizer, config)

if __name__ == "__main__":
    main()
