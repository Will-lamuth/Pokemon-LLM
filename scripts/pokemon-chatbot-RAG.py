import requests
import json
import re

# Load Pokémon database with Gen 1 & 2 movesets
with open("pokemon_gen12_moveset.json") as f:
    pokemon_db = json.load(f)

# Extract Pokémon names from generated list
def extract_pokemon_names_from_text(text):
    return [name for name in pokemon_db if name.lower() in text.lower()]

# Retrieve movesets and build prompt
def build_grounded_prompt(pokemon_list, user_request):
    prompt = "You are a Pokémon expert. Only use Gen 1 and 2 Pokémon and moves.\n\n"
    for name in pokemon_list:
        moves = pokemon_db.get(name, {}).get("legal_moves", [])
        prompt += f"{name} can use: {', '.join(moves)}\n"
    prompt += f"\nUser: {user_request.strip()}"
    return prompt

# Step 1 prompt to generate team
def ask_llm_to_pick_team(user_prompt):
    step1_prompt = f"You are a Pokémon expert. Pick 6 Pokémon from Gen 1 or 2 to build a competitive Smogon-style team.\n\nUser: {user_prompt.strip()}"
    payload = {"model": "pokemon-assistant-v3", "messages": [{"role": "user", "content": step1_prompt}]}
    response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, stream=True)

    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    full_response += content
                except json.JSONDecodeError:
                    continue  # skip malformed lines
        return full_response.strip()
    else:
        print(f"❌ Error: {response.status_code}")
        return ""


# Step 2 prompt with grounded moves
def ask_llm_to_build_team(grounded_prompt):
    payload = {"model": "pokemon-assistant-v3", "messages": [{"role": "user", "content": grounded_prompt}]}
    response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, stream=True)

    if response.status_code == 200:
        print("\nAssistant:", end=" ", flush=True)
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    content = json_data.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    full_response += content
                except json.JSONDecodeError:
                    print(f"\n[Error parsing line: {line}]")
        print()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

# --- MAIN LOOP ---
print("Pokémon Assistant LLM (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # STEP 1: Let LLM pick team if no Pokémon given
    print("\n📥 Asking LLM to generate Pokémon for your team...")
    generated_team_text = ask_llm_to_pick_team(user_input)
    print(f"\n📤 LLM Selected: {generated_team_text}")

    # Extract Pokémon names from its answer
    selected_pokemon = extract_pokemon_names_from_text(generated_team_text)
    print(f"\n✅ Pokémon Detected: {', '.join(selected_pokemon)}")

    # STEP 2: Build grounded prompt with correct moves
    grounded_prompt = build_grounded_prompt(selected_pokemon, user_input)
    print("\n📥 Injected Prompt Sent to LLM:\n" + grounded_prompt + "\n")

    # Ask LLM to now build the full team with movesets
    ask_llm_to_build_team(grounded_prompt)

