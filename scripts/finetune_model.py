import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# --- Config ---
model_name = "unsloth/llama-2-7b-bnb-4bit"  # Public Unsloth-compatible model
data_path = "../scripts/pokemon_learnsets_chat_format.jsonl"
output_dir = "unsloth-llama3-pokemon"
max_seq_length = 512

# --- Load model and tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
)

# --- Add special tokens BEFORE training ---
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
added = tokenizer.add_special_tokens(special_tokens)
if added > 0:
    print(f"🔧 Added {added} special tokens to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

# --- Apply LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ]
)

# --- Load and process dataset ---
dataset = load_dataset("json", data_files=data_path)["train"]

def format_prompt(example):
    try:
        user_message = next(m["content"] for m in example["messages"] if m["role"] == "user")
        assistant_message = next(m["content"] for m in example["messages"] if m["role"] == "assistant")
    except Exception as e:
        print(f"⚠️ Skipping malformed example: {e}")
        return {"text": ""}

    return {
        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant_message}<|eot_id|>"
    }



dataset = dataset.map(format_prompt)

# --- Tokenize dataset ---
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

tokenized = dataset.map(tokenize_fn, batched=True)

# --- Training args ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,  # More epochs for small dataset
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# --- Train ---
trainer.train()

# --- Save LoRA adapter only ---
model.save_pretrained(f"{output_dir}/lora")
tokenizer.save_pretrained(f"{output_dir}/lora")

# --- Merge LoRA adapter into base model ---
print("🔄 Merging LoRA adapter into base model...")
model = model.merge_and_unload()

# --- Save full merged model ---
merged_output_dir = f"{output_dir}/merged"
model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)

print(f"✅ Merged model saved to: {merged_output_dir}")
