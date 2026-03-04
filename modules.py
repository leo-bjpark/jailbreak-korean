import torch
from tqdm import tqdm

def generate_response(model, tokenizer, prompts, batch_size=8, max_new_tokens=128):
    all_decoded_outputs = []

    # 1. Iterate through the prompts in chunks (batches)
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i : i + batch_size]
        
        # 2. Tokenize the current batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # 3. Generate outputs with gradient calculation disabled for efficiency
        with torch.no_grad():
            outputs = model.generate(
                **inputs,  # Unpacks input_ids and attention_mask automatically
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 4. Decode only the new tokens (removing the prompt)
        input_len = inputs.input_ids.shape[1]
        batch_outputs = [
            tokenizer.decode(output[input_len:], skip_special_tokens=True)
            for output in outputs
        ]
        
        all_decoded_outputs.extend(batch_outputs)

    return all_decoded_outputs

def format_chat(
    tokenizer,
    prompt: str,
    response: str | None = None,
    add_generation_prompt: bool = False,
):
    """
    Chat Template for the model.
    """

    messages = [{"role": "user", "content": prompt}]

    if response is not None:
        messages.append(
            {"role": "assistant", "content": response}
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    
    
def parse_korean_tag(response: str) -> str:
    # "> 오늘 날씨 맑은 </k>" -> "오늘 날씨 맑은"
    # > location:
    index = response.find(">")
    if index == -1:
        return None 
    response = response[index + 1 :]
    index = response.find("</k>")
    if index == -1:
        return None 
    response = response[:index]
    return response.strip()