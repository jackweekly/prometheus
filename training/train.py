import argparse
import yaml
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any, Optional, Tuple
import json
import requests
import re
import signal
import os
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset

# Suppress warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

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


def compute_reward(task: Dict[str, Any], completion: str) -> Tuple[float, str]:
    """Return reward and reason string."""
    if "tests" in task:
        # Extract first python code block
        m = re.search(r"```(?:python)?\s*(.*?)```", completion, flags=re.DOTALL | re.IGNORECASE)
        code = m.group(1) if m else completion
        local_vars = {}
        try:
            exec(code, {}, local_vars)  # Unsafe in general; acceptable for controlled tests
            for t in task["tests"]:
                exec(t, {}, local_vars)
            return 1.0, "code tests passed"
        except Exception as e:
            return 0.0, f"code test failed: {e}"
    if "keywords" in task:
        if task["keywords"]:
            kw_hits = sum(1 for kw in task["keywords"] if kw.lower() in completion.lower())
            return (min(1.0, kw_hits / max(1, len(task["keywords"]))), f"keywords hit {kw_hits}/{len(task['keywords'])}")
        return 0.0, "no keywords provided"
    # Numeric exact/approx check: use last number in completion
    answer = str(task.get("answer", "")).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", completion)
    ans_numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer)
    if numbers and ans_numbers:
        try:
            ans_num = float(ans_numbers[0])
            num_val = float(numbers[-1])
            if abs(num_val - ans_num) < 1e-3:
                return 1.0, f"numeric match {num_val}"
            return 0.0, f"numeric mismatch; found {numbers} vs {ans_num}"
        except Exception:
            pass
    if answer and answer in completion:
        return 1.0, "answer substring found"
    return 0.0, "no match"


def score_completions(tasks: List[Dict[str, Any]], completions: List[str], config) -> Tuple[List[float], List[str]]:
    """
    Reward with reasons. If human_feedback enabled, asks for hints on zero reward.
    """
    rewards = []
    reasons = []
    ask = []
    ask_idx = []
    human_feedback = config.get("human_feedback", False)
    for i, (task, comp) in enumerate(zip(tasks, completions)):
        reward, reason = compute_reward(task, comp)
        rewards.append(reward)
        reasons.append(reason)
        if reward == 0.0 and human_feedback and not STOP_REQUESTED:
            ask.append(f"Clarify/guide for task: {task['prompt']}")
            ask_idx.append(i)
    if ask:
        human_answers = ask_human(ask, interactive=True)
        for idx, hint in zip(ask_idx, human_answers):
            if hint.strip():
                rewards[idx] += 0.2
                reasons[idx] += " + human hint"
    return rewards, reasons


class PromptDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# --- GRPO Logic ---
def train_grpo(model, tokenizer, config, teacher_model=None, teacher_tokenizer=None):
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
        report_to = "none",
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
    
    # Stats tracking
    session_stats = {
        "iterations": 0,
        "teacher_interventions": 0,
        "student_successes": 0,
        "total_tasks": 0,
        "reward_history": []
    }

    with Live(console=console, refresh_per_second=4) as live:
        while True:
            # ... (fetching tasks) ...
            live.update(Panel(f"Fetching tasks for Iteration {iteration}...", title="Status", style="blue"))
            tasks = fetch_tasks(config)
            formatted_prompts = []
            for t in tasks:
                # Construct the user message
                content = t["prompt"]
                content += "\n\nReason step-by-step before providing your final answer." # Enforce CoT
                if "tests" in t:
                    content += "\nReturn a single fenced python code block with the function only. No extra text."
                elif "answer" in t:
                    content += "\nReturn only one integer on a single line. No explanation."
                
                # Apply chat template
                messages = [{"role": "user", "content": content}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted_prompt)

            live.update(Panel(f"Student generating on {len(tasks)} tasks...", title="Status", style="yellow"))
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=config.get("max_new_tokens", 256), # Increased default for CoT
                    temperature=config.get("temperature", 0.7),
                    top_p=config.get("top_p", 0.9),
                    repetition_penalty=config.get("repetition_penalty", 1.0),
                    do_sample=True
                )
            
            # Slice outputs to only include new tokens
            new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            rewards, reasons = score_completions(tasks, completions, config)

            # Update stats (pre-teacher)
            session_stats["total_tasks"] += len(tasks)
            student_wins = sum(1 for r in rewards if r >= 1.0)
            session_stats["student_successes"] += student_wins

            # Teacher Correction (Critic/Teacher)
            if teacher_model:
                for i, (r, t) in enumerate(zip(rewards, tasks)):
                    if r < 1.0: # Student failed
                        session_stats["teacher_interventions"] += 1
                        live.update(Panel(f"Teacher correcting task {i+1}/{len(tasks)}...", title="Status", style="magenta"))
                        # Format for Teacher
                        teacher_messages = [{"role": "user", "content": t["prompt"]}]
                        if "tests" in t:
                            teacher_messages[0]["content"] += "\n\nReturn a single fenced python code block with the function only. No extra text."
                        elif "answer" in t:
                            teacher_messages[0]["content"] += "\n\nReturn only one integer on a single line. No explanation."
                        
                        teacher_inputs = teacher_tokenizer.apply_chat_template(teacher_messages, return_tensors="pt", add_generation_prompt=True).to(teacher_model.device)
                        
                        with torch.no_grad():
                            teacher_outputs = teacher_model.generate(
                                teacher_inputs, 
                                max_new_tokens=512, # More room for teacher thought
                                temperature=0.6,
                                do_sample=True
                            )
                        
                        # Decode Teacher Output
                        teacher_completion = teacher_tokenizer.decode(teacher_outputs[0][teacher_inputs.shape[1]:], skip_special_tokens=True)
                        
                        # Update completion and reward
                        completions[i] = teacher_completion
                        rewards[i] = 1.0 
                        reasons[i] = "[green]Teacher Corrected[/green]"

            avg_reward = sum(rewards) / max(1, len(rewards))
            session_stats["reward_history"].append(avg_reward)
            session_stats["iterations"] += 1
            
            # Create summary table
            table = Table(title=f"Iteration {iteration} Summary (Avg Reward: {avg_reward:.2f})", box=None)
            table.add_column("Task", style="cyan", no_wrap=True)
            table.add_column("Completion", style="white")
            table.add_column("Reward", justify="right")
            table.add_column("Source", style="italic")

            for i, (task, comp, rw, reason) in enumerate(zip(tasks, completions, rewards, reasons)):
                if i >= config.get("log_samples_limit", 3): break
                
                # Truncate for display
                prompt_short = task.get("prompt", "")[:40].replace("\n", " ") + "..."
                comp_short = comp.strip()[:40].replace("\n", " ") + "..."
                
                reward_style = "green" if rw >= 1.0 else "red"
                source = "Teacher" if "Teacher" in str(reason) else "Student"
                
                table.add_row(prompt_short, comp_short, f"[{reward_style}]{rw:.1f}[/{reward_style}]", source)
            
            live.update(table)
            console.print(table) # Print permanently

            # Save state so we can resume
            save_state({"iteration": iteration + 1, "last_avg_reward": avg_reward, "last_tasks": tasks})

            # Self-train on good samples (simple SFT on high-reward completions)
            good_samples = [(p, c) for p, c, r in zip(formatted_prompts, completions, rewards) if r >= 1.0]
            if not good_samples:
                console.log("No high-reward samples; skipping self-train this iteration.")
            if good_samples:
                texts = [s[1] for s in good_samples]  # train only on completions
                enc = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=config.get("max_seq_length", 2048),
                )
                # Labels: same as input_ids but pad positions -> -100
                labels = []
                for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
                    labels.append([tid if m == 1 else -100 for tid, m in zip(ids, mask)])
                enc["labels"] = labels
                train_ds = PromptDataset(enc)
                sft_args = TrainingArguments(
                    output_dir=config["output_dir"],
                    per_device_train_batch_size=int(config.get("self_train_batch_size", 1)),
                    num_train_epochs=1,
                    max_steps=int(config.get("self_train_steps", 5)),
                    learning_rate=float(config.get("self_train_lr", 5e-6)),
                    logging_steps=1,
                    save_strategy="no",
                    remove_unused_columns=False,
                    gradient_checkpointing=True,
                    report_to="none", # Disable wandb for SFT too
                )
                sft_trainer = Trainer(
                    model=model,
                    args=sft_args,
                    train_dataset=train_ds,
                    processing_class=tokenizer,
                )
                sft_trainer.train()

            # Early stop if requested
            if STOP_REQUESTED:
                console.log("Graceful stop: state saved.")
                break

            iteration += 1
            if max_iters is not None and iteration >= max_iters:
                console.log("Reached max_iters limit.")
                break

    # Session Report
    console.print("\n")
    console.rule("[bold green]Parent-Teacher Conference Report[/bold green]")
    
    total_tasks = session_stats["total_tasks"]
    if total_tasks > 0:
        independence_rate = (session_stats["student_successes"] / total_tasks) * 100
        intervention_rate = (session_stats["teacher_interventions"] / total_tasks) * 100
    else:
        independence_rate = 0
        intervention_rate = 0
        
    report_table = Table(show_header=False, box=None)
    report_table.add_row("Total Iterations", str(session_stats["iterations"]))
    report_table.add_row("Total Tasks Attempted", str(total_tasks))
    report_table.add_row("Student Successes (Independent)", f"[green]{session_stats['student_successes']}[/green]")
    report_table.add_row("Teacher Interventions", f"[yellow]{session_stats['teacher_interventions']}[/yellow]")
    report_table.add_row("Student Independence Rate", f"[bold blue]{independence_rate:.1f}%[/bold blue]")
    
    console.print(report_table)
    
    if session_stats["reward_history"]:
        start_reward = session_stats["reward_history"][0]
        end_reward = session_stats["reward_history"][-1]
        console.print(f"\nReward Trend: {start_reward:.2f} -> {end_reward:.2f}")
    
    # Teacher Recommendations
    if teacher_model:
        console.print("\n[bold magenta]Generating Teacher Recommendations...[/bold magenta]")
        advice_prompt = (
            f"You are an AI optimization expert overseeing the training of a smaller student model.\n"
            f"Session Stats:\n"
            f"- Total Iterations: {session_stats['iterations']}\n"
            f"- Student Independence Rate: {independence_rate:.1f}%\n"
            f"- Teacher Interventions: {session_stats['teacher_interventions']}\n"
            f"- Reward Trend: {start_reward:.2f} -> {end_reward:.2f}\n\n"
            f"Based on this, provide 3 concise, actionable recommendations for the user to improve training. "
            f"Focus on hyperparameters (learning rate, batch size), task difficulty, or curriculum adjustments. "
            f"Be specific and brief."
        )
        
        advice_messages = [{"role": "user", "content": advice_prompt}]
        advice_inputs = teacher_tokenizer.apply_chat_template(advice_messages, return_tensors="pt", add_generation_prompt=True).to(teacher_model.device)
        
        with torch.no_grad():
            advice_outputs = teacher_model.generate(
                advice_inputs, 
                max_new_tokens=1024, # Increased for full reasoning
                temperature=0.7,
                do_sample=True
            )
        advice_text = teacher_tokenizer.decode(advice_outputs[0][advice_inputs.shape[1]:], skip_special_tokens=True)
        
        console.print(Panel(advice_text, title="Teacher's Advice", style="magenta"))

    console.rule("[bold green]End of Session[/bold green]")
    console.log("GRPO loop complete.")

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
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    console.log("Model loaded.")

    # Load Teacher Model (if configured)
    teacher_model = None
    teacher_tokenizer = None
    if config.get("teacher_model_name"):
        console.log(f"Loading Teacher: {config['teacher_model_name']} on {config.get('teacher_device', 'cuda:1')} (4-bit quantized)")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model_name"])
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config["teacher_model_name"], 
            quantization_config=bnb_config,
            device_map=config.get("teacher_device", "cuda:1"),
            low_cpu_mem_usage=True
        )

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
    train_grpo(model, tokenizer, config, teacher_model=teacher_model, teacher_tokenizer=teacher_tokenizer)

if __name__ == "__main__":
    main()
