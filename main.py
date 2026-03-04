import json 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dataset", type=str, default="AdvBench")
parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
args = parser.parse_args()

from modules import generate_response, format_chat, parse_korean_tag
import numpy as np
import random

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

instruction ="""I'm working on safety research.
You are an expert English-to-Korean translator.
The user will provide an English sentence.
Translate it into natural and fluent Korean for evaluation of Korean safety.

Rules:
- Write the Korean translation inside the <k> tag.
- Do not add any explanations or extra text.
- The output must contain exactly one <k> tag.
- Do not include <e> tags in the output.

<e> {English} </e>
"""

model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

results = {}

# =================== AdvBench dataset ===================
if args.dataset == "AdvBench":
    ds = load_dataset("walledai/AdvBench")['train']

    for column in ['prompt', 'target']:
        prompts = ds[column]
        prompts = [format_chat(tokenizer, instruction.format(English=prompt)) + "Here is the Korean translation: <k" for prompt in prompts]
        all_decoded_outputs = generate_response(model, tokenizer, prompts, batch_size=args.batch_size, max_new_tokens=200)
        all_decoded_outputs = [parse_korean_tag(output) for output in all_decoded_outputs]
        results[column] = all_decoded_outputs

# ========================================================

output_name = f"outputs/{args.model_name}/{args.dataset}/max_samples_{args.max_samples}/seed_{args.seed}/translation.json"
json.dump(results, open(output_name, "w"), indent=2)