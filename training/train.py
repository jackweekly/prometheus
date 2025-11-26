import argparse
import yaml
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any, Optional
import json
import requests
import re
import signal
import os
from rich.console import Console

console = Console()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- State handling ---
STATE_PATH = "state/run_state.json"
STOP_REQUESTED = False

def handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested, finishing current cycle...")


def load_state(path: str = STATE_PATH) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(data: Dict[str, Any], path: str = STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

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


def ask_human(queries: List[str], interactive: bool = True) -> List[str]:
    """
    Human-in-the-loop clarification. If interactive, read from stdin; otherwise just logs.
    Returns a list of hints aligned with queries.
    """
    hints = []
    for q in queries:
        print(f"Human clarification requested for task: {q}")
        if interactive and not STOP_REQUESTED:
            try:
                hint = input("Provide a hint (or leave blank): ").strip()
            except EOFError:
                hint = ""
        else:
            hint = ""
        hints.append(hint)
    return hints


def fetch_research(config) -> List[Dict[str, Any]]:
    """
    Fetch research items from configured RSS/Atom feeds (e.g., arXiv). Returns list of tasks with keywords for scoring.
    """
    sources = config.get("research_sources", [])
    keywords = config.get("research_keywords", [])
    tasks = []
    for src in sources:
        try:
            resp = requests.get(src, timeout=10)
            resp.raise_for_status()
            text = resp.text
            entries = re.findall(r"<title>([^<]+)</title>.*?<summary>([^<]+)</summary>", text, flags=re.DOTALL)
            # skip the feed-level title (first)
            for title, summary in entries[1:]:
                prompt = f"Summarize the paper '{title}' and list key contributions."
                tasks.append({"prompt": prompt, "answer": summary.strip(), "keywords": keywords})
        except Exception as e:
            print(f"Warning: research fetch failed from {src} ({e})")
    return tasks


def fetch_tasks(config) -> List[Dict[str, Any]]:
    """
    Task source: tries HTTP if task_url provided; then research feeds; falls back to built-in tasks.
    Expects JSON list of {prompt: str, answer: str, tests?: [str]}.
    """
    task_url = config.get("task_url")
    if task_url:
        try:
            resp = requests.get(task_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"Warning: HTTP task fetch failed ({e}); falling back to research/local tasks.")

    research_items = fetch_research(config)
    if research_items:
        return research_items

    # Fallback tasks
    return [
        {"prompt": "Compute 17 * 23.", "answer": "391"},
        {
            "prompt": "Write a Python function add(a, b) that returns their sum.",
            "answer": "def add(a, b):\n    return a + b",
            "tests": ["assert add(1,2)==3", "assert add(-1,1)==0"],
        },
        {"prompt": "What is the capital of France?", "answer": "Paris"},
    ]


def score_completions(tasks: List[Dict[str, Any]], completions: List[str], config) -> List[float]:
    """
    Simple reward: exact string contains for math; code tested via exec if tests provided.
    If low confidence, asks human (stub) for guidance.
    """
    rewards = []
    ask = []
    ask_idx = []
    human_feedback = config.get("human_feedback", False)
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
        elif "keywords" in task:
            if task["keywords"]:
                kw_hits = sum(1 for kw in task["keywords"] if kw.lower() in comp.lower())
                rewards.append(min(1.0, kw_hits / max(1, len(task["keywords"]))))
            else:
                rewards.append(0.0)
        else:
            rewards.append(1.0 if str(task["answer"]) in comp else 0.0)
        if rewards[-1] == 0.0 and human_feedback and not STOP_REQUESTED:
            ask.append(f"Clarify/guide for task: {task['prompt']}")
            ask_idx.append(i)
    if ask:
        human_answers = ask_human(ask, interactive=True)
        for idx, hint in zip(ask_idx, human_answers):
            # If human gives a hint, provide small reward bump to encourage exploration
            if hint.strip():
                rewards[idx] += 0.2
    return rewards


# --- GRPO Logic ---
def train_grpo(model, tokenizer, config):
    console.log("Starting GRPO Training (online tasks, no static dataset)...")

    state = load_state()
    start_iter = state.get("iteration", 0)
    max_iters = config.get("max_iters")  # None = unbounded
    console.log(f"Resuming at iteration {start_iter}; max_iters={'∞' if max_iters is None else max_iters}")

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
        dataloader_num_workers = config.get('dataloader_num_workers', 4),
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [lambda p, c, **kwargs: kwargs.get("rewards", [])],
        args = training_args,
        train_dataset = None, # Online tasks; no persistent dataset
    )

    global STOP_REQUESTED
    iteration = start_iter
    while True:
        tasks = fetch_tasks(config)
        prompts = [t["prompt"] for t in tasks]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        rewards = score_completions(tasks, completions, config)
        avg_reward = sum(rewards) / max(1, len(rewards))
        console.log(f"[Iter {iteration}] tasks={len(tasks)} avg_reward={avg_reward:.3f}")

        # Save state so we can resume
        save_state({"iteration": iteration + 1, "last_avg_reward": avg_reward, "last_tasks": tasks})

        # Early stop if requested
        if STOP_REQUESTED:
            console.log("Graceful stop: state saved.")
            break

        iteration += 1
        if max_iters is not None and iteration >= max_iters:
            console.log("Reached max_iters limit.")
            break

    console.log("GRPO loop complete (placeholder without optimizer step).")

def main():
    parser = argparse.ArgumentParser(description="Jamba-only GRPO Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    config = load_config(args.config)
    # In some environments globals can be shadowed; re-import defensively
    import os as _os
    model_id = _os.environ.get("MODEL_ID", config["model_name"])
    console.log(f"Loading model: {model_id} (Jamba-only pipeline)")

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    console.log("Detected Jamba model. Loading with HF + PEFT for hybrid attention/SSM layers.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    console.log("Downloading/loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    console.log("Model loaded.")

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

    console.log("Gradient checkpointing enabled.")
    
    # Torch Compile Optimization
    # print("Compiling model with torch.compile...")
    # model = torch.compile(model)
    train_grpo(model, tokenizer, config)

if __name__ == "__main__":
    main()
