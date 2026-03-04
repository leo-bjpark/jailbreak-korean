import json 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dataset", type=str, default="AdvBench")
parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
args = parser.parse_args()

from modules import generate_response, format_chat, parse_korean_tag, instruction_prompt, assistant_prompt
import numpy as np
import random
import os 

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)



model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

results = {}
config = {}
# =================== AdvBench dataset ===================
if args.dataset == "AdvBench":
    ds = load_dataset("walledai/AdvBench")['train']

    for column in ['prompt', 'target']:
        prompts = ds[column]
        instruction = instruction_prompt if column == "prompt" else assistant_prompt
        prompts = [format_chat(tokenizer, instruction.format(English=prompt)) + "Here is the Korean translation: <k" for prompt in prompts]
        all_decoded_outputs = generate_response(model, tokenizer, prompts, batch_size=args.batch_size, max_new_tokens=200)
        all_decoded_outputs = [parse_korean_tag(output) for output in all_decoded_outputs]
        results[column] = all_decoded_outputs
        config[column] = sum(1 for output in all_decoded_outputs if output is not None) / len(all_decoded_outputs)
        config[column] = f"{config[column]:.2%}"
# ========================================================
# Save Results 
output_dir = f"outputs/{args.model_name}/{args.dataset}/seed_{args.seed}"
os.makedirs(output_dir, exist_ok=True)
json.dump(results, open(f"{output_dir}/translation.json", "w"), indent=2, ensure_ascii=False)
json.dump(config, open(f"{output_dir}/config.json", "w"), indent=2, ensure_ascii=False)