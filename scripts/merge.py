import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
model_path = "unsloth-llama3-pokemon/merged"
prompt = "Which moves can Charizard learn?"

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --- Ensure special tokens exist ---
special_tokens = {
    "eos_token": "<|eot_id|>",
    "pad_token": "<|eot_id|>",
    "additional_special_tokens": [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ]
}

# Add special tokens only if they are missing
added = tokenizer.add_special_tokens(special_tokens)
if added > 0:
    print(f"🔧 Added {added} special tokens to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

# --- Format prompt to match training style ---
input_text = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    f"{prompt}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

# --- Tokenize and run generation ---
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
    )

# --- Decode and extract the assistant's response ---
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n"
if assistant_prefix in decoded:
    response = decoded.split(assistant_prefix)[-1].split("<|eot_id|>")[0].strip()
else:
    response = "(No assistant response found.)"

# --- Output ---
print("\n🧠 Assistant Response:\n")
print(response)
