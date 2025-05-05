### Setup
```
create dev environment:
- conda create poke-llm
- conda activate poke-llm
- pip install -r requirements.txt

View Models:
- ollama list

Create Models with modelfile:
- ollama create randommodel -f Modelfile

Finetuneing Model:
1. use finetune_model.py to perform LoRA finetune with model on .JSON dataset. Script will output a lora file
2. Merge the finetuned lora file with the base model to create the finetuned model. This will output a merged folder
3. Navigate to llama.cpp and use the script 'convert_hf_to_gguf_update.py' with command - python3 convert_hf_to_gguf_update.py --outtype f16 --outfile ./your-model.gguf ./merged-model
```
