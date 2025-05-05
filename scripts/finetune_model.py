# This is a script to (test)finetune TinyLLama on a basic pokemon .jsonl dataset with LoRA
# outputs: ./tinyllama-pokemon-lora

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# ---- Load Model ----
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

# ---- Prepare for LoRA Training ----
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ---- Load Dataset ----
data = load_dataset("json", data_files="scripts/pokemon_finetune_full.jsonl")

# Format for instruction tuning
def format_prompt(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"
    }

tokenized = data.map(format_prompt)
tokenized = tokenized.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# ---- Training Arguments ----
training_args = TrainingArguments(
    output_dir="./tinyllama-pokemon-finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# ---- Save LoRA Model ----
model.save_pretrained("tinyllama-pokemon-lora")
tokenizer.save_pretrained("tinyllama-pokemon-lora")