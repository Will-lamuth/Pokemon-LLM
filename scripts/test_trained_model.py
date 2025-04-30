from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_path = "./tinyllama-pokemon-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, lora_model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    prompt = input("🧠 Ask something about Pokémon: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        break
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    response = pipe(formatted, max_new_tokens=150, do_sample=True, temperature=0.8)
    print(response[0]['generated_text'].split('<|assistant|>\n')[1])