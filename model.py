from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

# Step 1: Load the Pokémon Data
pokemon_data = pd.read_csv('/mnt/data/Pokemon.csv')

# Convert the Pokémon data into text descriptions
pokemon_texts = []
for index, row in pokemon_data.iterrows():
    # Format the Pokémon data into a text string
    pokemon_description = f"{row['Name']} is a {row['Type 1']} and {row['Type 2'] if pd.notna(row['Type 2']) else 'no'}-type Pokémon with a total stat of {row['Total']}. It has {row['HP']} HP, {row['Attack']} Attack, {row['Defense']} Defense, {row['Sp. Atk']} Special Attack, {row['Sp. Def']} Special Defense, and {row['Speed']} Speed. It is from Generation {row['Generation']} and is {'a Legendary Pokémon' if row['Legendary'] else 'not a Legendary Pokémon'}."
    pokemon_texts.append(pokemon_description)

# Step 2: Tokenize the Data
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if the tokenizer has a pad_token, if not, set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the Pokémon descriptions
inputs = tokenizer(pokemon_texts, return_tensors="pt", padding=True, truncation=True)

# Step 3: Load the Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically selects CPU or GPU/MPS on your system
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

# Step 4: Use Accelerator to handle the device automatically
accelerator = Accelerator()

# Prepare the model and inputs using Accelerator
model, inputs = accelerator.prepare(model, inputs)

# Step 5: Run Inference to Generate Output
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,  # Adjust the number of tokens to generate
        do_sample=True,  # If you want more randomness in the generated text
        top_k=50,  # Controls randomness (higher values = more randomness)
        top_p=0.95  # Nucleus sampling for randomness
    )

# Decode the output to get the response as text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(response)

# Optional: Step 6 - Fine-Tuning the Model (this step requires significant resources)
# Convert your dataset into a HuggingFace Dataset
pokemon_df = pd.DataFrame(pokemon_texts, columns=["text"])
pokemon_dataset = Dataset.from_pandas(pokemon_df)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluate after each epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=4,   # batch size per device
    num_train_epochs=3,              # number of training epochs
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pokemon_dataset,
)

# Fine-tune the model (this step requires a lot of resources)
trainer.train()

# Step 7: Saving the Model (Optional)
# Save the fine-tuned model for future use
model.save_pretrained("./pokemon_model")
tokenizer.save_pretrained("./pokemon_model")

# Step 8: Testing the Trained Model
# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./pokemon_model")
tokenizer = AutoTokenizer.from_pretrained("./pokemon_model")

# Generate response for a custom prompt
prompt = "Suggest a balanced Gen 9 OU team that can handle Iron Valiant and Dragapult."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

# Decode and print the result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
