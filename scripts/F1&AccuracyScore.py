import requests
import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Pokémon database with Gen 1 & 2 movesets
with open("pokemon_gen12_moveset.json") as f:
    pokemon_db = json.load(f)

# Extract (pokemon, move) pairs from the LLM's generated text
def extract_movesets_from_text(text):
    moveset = {}
    current_pokemon = None
    move_pattern = re.compile(r"\+\s+(.*)", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()

        # Match Pokémon header line
        pokemon_match = re.match(r"^\d+\.\s+\*\*(.*?)\*\*", line)
        if pokemon_match:
            current_pokemon = pokemon_match.group(1)
            moveset[current_pokemon] = []
            continue

        # Match moves
        move_match = move_pattern.match(line)
        if move_match and current_pokemon:
            move = move_match.group(1).strip()
            moveset[current_pokemon].append(move)

    return moveset

# Evaluate move legality
def evaluate_movesets(moveset_dict):
    y_true = []
    y_pred = []

    for pokemon, moves in moveset_dict.items():
        legal_moves = pokemon_db.get(pokemon, {}).get("legal_moves", [])
        legal_set = set([m.lower() for m in legal_moves])

        for move in moves:
            y_true.append(1)  # 1 = expected to be legal
            y_pred.append(1 if move.lower() in legal_set else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

# Ask LLM to generate a team
def ask_llm_to_pick_team(user_prompt):
    step1_prompt = f"You are a Pokémon expert. Pick 6 Pokémon from Gen 1 or 2 to build a competitive Smogon-style team with movesets.\n\nUser: {user_prompt.strip()}"
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
                    continue
        return full_response.strip()
    else:
        print(f"❌ Error: {response.status_code}")
        return ""

# --- MAIN LOOP ---
print("Pokémon Assistant LLM — Moveset Evaluation Only (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    print("\n📥 Asking LLM to generate a full team with moves...")
    llm_output = ask_llm_to_pick_team(user_input)
    print(f"\n📤 LLM Response:\n{llm_output}")

    # Extract and evaluate movesets
    moveset = extract_movesets_from_text(llm_output)
    precision, recall, f1 = evaluate_movesets(moveset)

    print(f"\n📊 Moveset Evaluation — Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
