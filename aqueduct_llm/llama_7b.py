import os
import time

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class Config:
    def __init__(self):
        self.model_path = "aleksickx/llama-7b-hf"
        self.device = "cuda"
    
    def describe(self) -> str:
        print("Running LLaMA 7B with the following config:")
        attrs = {
            "model_path": self.model_path,
            "device": self.device,
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

    tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
    model = LlamaForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16).to(config.device)
    print("Finished loading model.")
    end_time = time.time()
    time_taken = end_time - start_time

    print(f'Time taken: {time_taken:.5f} seconds')

    results = []
    for message in messages:
        batch = tokenizer(
            message,
            return_tensors="pt", 
            add_special_tokens=False
        )

        batch = {k: v.to(config.device) for k, v in batch.items()}
        generated = model.generate(batch["input_ids"], max_length=100)

        results.append(tokenizer.decode(generated[0]))

    return results[0] if len(results) == 1 else results