import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel

def calculate_entropy_varentropy(logits):
    """
    Calculates entropy and varentropy from logits.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy: H(X) = -sum(p(x) * log(p(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Varentropy: V(X) = sum(p(x) * (log(p(x)) + H(X))^2)
    varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1))**2, dim=-1)
    
    return entropy, varentropy

def entropix_generate(model, tokenizer, prompt, max_new_tokens=128):
    """
    Generates text using Entropix sampling strategy.
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    
    generated_ids = input_ids
    
    print(f"Prompt: {prompt}")
    print("-" * 20)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            entropy, varentropy = calculate_entropy_varentropy(next_token_logits)
            
            # Thresholds (tunable)
            entropy_threshold = 2.0 
            varentropy_threshold = 5.0
            
            current_entropy = entropy.item()
            current_varentropy = varentropy.item()
            
            # Decision Logic
            if current_entropy < entropy_threshold:
                # Low Entropy -> Confident -> Greedy Sampling
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                strategy = "Greedy"
            elif current_varentropy > varentropy_threshold:
                # High Entropy + High Varentropy -> Confused/Complex -> Insert "Thinking"
                # In a full implementation, we might branch or insert a CoT prompt.
                # Here we simulate it by sampling with a lower temperature to focus.
                probs = F.softmax(next_token_logits / 0.7, dim=-1) # Low temp
                next_token = torch.multinomial(probs, num_samples=1)
                strategy = "Thinking (Low Temp)"
            else:
                # High Entropy + Low Varentropy -> Uncertain but flat -> Standard Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                strategy = "Sampling"
            
            # print(f"E: {current_entropy:.2f}, V: {current_varentropy:.2f} -> {strategy}")
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    # Example usage
    model_name = "Qwen/Qwen2.5-3B-Instruct" # Or your trained model path
    print(f"Loading {model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    prompt = "Explain quantum entanglement in simple terms."
    output = entropix_generate(model, tokenizer, prompt)
    print(output)

if __name__ == "__main__":
    main()
