import time

import torch
from aqueduct_llm.utils.dolly_instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

class Config:
    def __init__(self):
        self.model_path = "databricks/dolly-v2-3b"
    
    def describe(self) -> str:
        print("Running Dolly V2 7B with the following config:")
        attrs = {
            "model_path": self.model_path,
        }
        print("\n".join([f"{attr}: {value}" for attr, value in attrs.items()]))

def generate(messages):
    config = Config()
    config.describe()

    if isinstance(messages, str):
        messages = [messages]
    elif isinstance(messages, list):
        if not all(isinstance(message, str) for message in messages):
            raise Exception("The elements in the list must be of type string.")
    else:
        raise Exception("Input must be a string or a list of strings.")

    if isinstance(messages, str):
        messages = [messages]

    print("Downloading and loading model...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(config.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    
    print("Finished loading model.")
    end_time = time.time()
    time_taken = end_time - start_time

    print(f'Time taken: {time_taken:.5f} seconds')

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    results = []
    for message in messages:
        res = generate_text(message)
        results.append(res[0]["generated_text"])

    return results[0] if len(results) == 1 else results