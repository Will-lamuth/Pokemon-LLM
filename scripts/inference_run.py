from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA 3 8B Instruct
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",      # Automatically selects CPU or MPS on Mac
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Your test prompt (Pokémon-style)
prompt = "Suggest a balanced Gen 9 OU team that can handle Iron Valiant and Dragapult."

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

# Decode and print the result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)