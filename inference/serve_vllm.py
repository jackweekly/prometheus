from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request
import uvicorn
import argparse
import json

# vLLM Server Implementation
# High-throughput serving using PagedAttention

app = FastAPI()
llm_engine = None
sampling_params = None

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])
    
    if not prompts:
        return {"error": "No prompts provided"}
    
    # Generate
    outputs = llm_engine.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)
        
    return {"generations": results}

def main():
    parser = argparse.ArgumentParser(description="vLLM Inference Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quantization", type=str, default=None, help="awq, gptq, squeezellm, or None")
    args = parser.parse_args()
    
    global llm_engine, sampling_params
    
    print(f"Initializing vLLM engine with model: {args.model}")
    llm_engine = LLM(model=args.model, quantization=args.quantization)
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    
    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
