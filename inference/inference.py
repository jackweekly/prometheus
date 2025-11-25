import argparse
from unsloth import FastLanguageModel
import torch

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model or HuggingFace ID")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--use_bitnet", action="store_true", help="Use BitNet optimization (requires bitnet.cpp for full speed)")
    args = parser.parse_args()

    if args.use_bitnet:
        print("Note: To run BitNet models efficiently, you should use bitnet.cpp.")
        print("See: https://github.com/microsoft/BitNet")
        # Fallback to standard loading for now if just testing logic
    
    print(f"Loading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        kv_cache_quantization = True, # Enable 8-bit KV cache
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer([args.prompt], return_tensors = "pt").to("cuda")

    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    print(tokenizer.batch_decode(outputs))

if __name__ == "__main__":
    main()
