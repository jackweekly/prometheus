import argparse
import json
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from datasets import load_dataset
import random

class HLEvaluator:
    """
    Evaluates a model on a subset of the Humanity's Last Exam (HLE) benchmark,
    using the MMLU dataset as a proxy.
    """
    def __init__(self, model, tokenizer, dataset_name="cais/mmlu", subset="anatomy"):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None

    def load_data(self, split="test"):
        """Loads the specified dataset and subset from the Hugging Face Hub."""
        print(f"Loading dataset: {self.dataset_name}, subset: {self.subset}")
        try:
            self.dataset = load_dataset(self.dataset_name, self.subset, split=split)
            print(f"Successfully loaded {len(self.dataset)} questions.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            self.dataset = []

    def format_prompt(self, item):
        """Formats a single item from the dataset into a multiple-choice prompt."""
        question = item["question"]
        choices = item["choices"]
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt

    def evaluate_single(self, item):
        """Evaluates the model on a single question."""
        prompt = self.format_prompt(item)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(**inputs, max_new_tokens=5, use_cache=True)
        prediction_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The model might just output 'A' or 'A.' or 'A. An answer'
        # We just check if the first character of the prediction is the correct one.
        prediction_char = prediction_text.strip().upper()[0]
        
        correct_choice_index = item["answer"]
        correct_char = chr(65 + correct_choice_index)
        
        is_correct = (prediction_char == correct_char)
        
        return is_correct, prediction_char, correct_char

    def run_evaluation(self, num_samples=10):
        """Runs the evaluation on a random sample of the dataset."""
        if not self.dataset:
            print("No dataset loaded. Aborting evaluation.")
            return

        print(f"Running evaluation on {num_samples} random samples...")
        
        correct_count = 0
        results = []
        
        # Take a random sample from the dataset
        sample_indices = random.sample(range(len(self.dataset)), k=num_samples)
        
        for i in tqdm(sample_indices):
            item = self.dataset[i]
            is_correct, pred, answ = self.evaluate_single(item)
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "question": item["question"],
                "prediction": pred,
                "answer": answ,
                "is_correct": is_correct
            })
            
        score = correct_count / num_samples
        
        report = {
            "score": score,
            "num_samples": num_samples,
            "results": results
        }
        
        # Save results to a file
        output_filename = f"hle_evaluation_results_{self.subset}.json"
        with open(output_filename, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"\nEvaluation complete. Score: {score:.2%}")
        print(f"Results saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate on Humanity's Last Exam (MMLU proxy)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--subset", type=str, default="anatomy")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    evaluator = HLEvaluator(model, tokenizer, subset=args.subset)
    evaluator.load_data()
    evaluator.run_evaluation(num_samples=args.num_samples)

if __name__ == "__main__":
    main()
