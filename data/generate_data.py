from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, EvolInstruct

# Evol-Instruct Implementation
# Reference: https://github.com/argilla-io/distilabel

def generate_synthetic_data():
    print("Initializing Distilabel pipeline with Evol-Instruct...")
    
    with Pipeline(name="evol-instruct-reasoning") as pipeline:
        # 1. Seed Data (Simple Instructions)
        load_dataset = LoadDataFromDicts(
            data=[
                {"instruction": "Solve for x: 2x + 5 = 15"},
                {"instruction": "Explain the theory of relativity."},
                {"instruction": "Write a python function to sort a list."},
                {"instruction": "What is the capital of France?"},
            ]
        )

        # 2. Evol-Instruct (Complicate the instructions)
        # This step uses a Teacher model to rewrite prompts to be more complex.
        evolve = EvolInstruct(
            llm=OpenAILLM(model="gpt-4o"),
            num_evolutions=2, # Evolve each prompt twice
            store_evolutions=True,
        )

        # 3. Generate Responses (Teacher model answers the complex prompts)
        generate = TextGeneration(
            llm=OpenAILLM(model="gpt-4o"),
        )

        # Connect steps
        load_dataset >> evolve >> generate

    print("Pipeline defined. Run pipeline.run() to execute (requires API keys).")
    # pipeline.run()

if __name__ == "__main__":
    generate_synthetic_data()

