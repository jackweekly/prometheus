import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def speculative_sampling(target_model, draft_model, tokenizer, prompt, max_new_tokens=128, gamma=4):
    """
    Speculative sampling using a draft model and a target model.
    gamma: number of tokens to draft in each step.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    
    print(f"Prompt: {prompt}")
    print(f"Draft Model: {draft_model.config._name_or_path}")
    print(f"Target Model: {target_model.config._name_or_path}")
    print("-" * 20)

    # Simplified loop for demonstration
    # In production, use vLLM's built-in speculative decoding or a highly optimized kernel
    
    generated_ids = input_ids
    
    for _ in range(max_new_tokens // gamma):
        # 1. Draft Step: Generate gamma tokens with draft model
        draft_outputs = draft_model.generate(
            generated_ids, 
            max_new_tokens=gamma, 
            do_sample=False, # Greedy for draft usually
            pad_token_id=tokenizer.eos_token_id
        )
        draft_tokens = draft_outputs[0, generated_ids.shape[1]:]
        
        # 2. Verification Step: Forward pass on target model with draft tokens
        # We check if the target model agrees with the draft tokens
        # For simplicity in this script, we just accept them if they match (greedy verification)
        # A real implementation calculates acceptance probability p(x)/q(x)
        
        with torch.no_grad():
            candidate_input = torch.cat([generated_ids, draft_tokens.unsqueeze(0)], dim=-1)
            target_outputs = target_model(candidate_input)
            target_logits = target_outputs.logits[:, -gamma-1:-1, :]
            target_preds = torch.argmax(target_logits, dim=-1)
            
        # Check matches
        matches = (target_preds[0] == draft_tokens)
        num_matches = matches.sum().item()
        
        # Append matched tokens
        if num_matches > 0:
            accepted_tokens = draft_tokens[:num_matches]
            generated_ids = torch.cat([generated_ids, accepted_tokens.unsqueeze(0)], dim=-1)
            
        # If mismatch, append the target model's actual next token at the mismatch position
        if num_matches < gamma:
            # The token at the first mismatch position from target model
            mismatch_idx = num_matches
            correct_token = torch.argmax(target_outputs.logits[:, -gamma-1+mismatch_idx, :], dim=-1)
            generated_ids = torch.cat([generated_ids, correct_token.unsqueeze(0)], dim=-1)
            
        # Stop if EOS
        if generated_ids[0, -1].item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    # Example usage
    target_model_name = "Qwen/Qwen2.5-3B-Instruct"
    draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    # Load Target (Large)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        load_in_4bit=True
    )
    
    # Load Draft (Small)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    
    prompt = "Write a poem about optimizing code."
    output = speculative_sampling(target_model, draft_model, tokenizer, prompt)
    print(output)

if __name__ == "__main__":
    main()
