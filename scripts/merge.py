# script to merge LoRA finetuned model with base model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "./tinyllama-pokemon-lora"

# Load base + LoRA
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, lora_path)

# Merge LoRA weights
model = model.merge_and_unload()

# Save as full Hugging Face model
model.save_pretrained("./tinyllama-pokemon-merged")

# Save tokenizer files
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained("./tinyllama-pokemon-merged")