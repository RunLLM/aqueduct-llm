import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import load_model, compute_skip_echo_len


@torch.inference_mode()
def generate(messages):
    model_path = "/home/ubuntu/vicuna/vicuna-7b/"
    device = "cuda"
    num_gpus = "1"
    max_gpu_memory = "13GiB"
    debug = False
    load_8bit = False

    if isinstance(messages, str):
        messages = [messages]
    elif isinstance(messages, list):
        if not all(isinstance(message, str) for message in messages):
            raise Exception("The elements in the list must be of type string.")
    else:
        raise Exception("Input must be a string or a list of strings.")

    if isinstance(messages, str):
        messages = [messages]

    results = []
    
    for message in messages:
        model, tokenizer = load_model(model_path, device,
            num_gpus, max_gpu_memory, load_8bit, debug=debug)

        msg = message

        conv = get_default_conv_template(model_path).copy()
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        skip_echo_len = compute_skip_echo_len(model_path, conv, prompt)
        outputs = outputs[skip_echo_len:]

        results.append(outputs)

    return results