import requests
import json
import re

# Load Pokémon database once
with open("pokemon_gen12_moveset.json") as f:
    pokemon_db = json.load(f)

# Extract moves from bullet-style lines (e.g., "- Thunderbolt")
def extract_moves(text):
    lines = text.splitlines()
    move_lines = []
    blocked_terms = {
        "atk", "def", "spa", "spd", "spe", "hp",
        "speed", "attack", "defense", "special attack", "special defense",
        "type", "nature", "moveset"
    }

    for line in lines:
        if re.match(r"^\s*[\-\+\u2022\*]\s*", line):  # bullets
            move_name = re.sub(r"^\s*[\-\+\u2022\*]\s*", "", line).strip()
            if move_name.lower() in blocked_terms:
                continue
            if re.match(r"^[A-Z][A-Za-z0-9 \-']+$", move_name) and ":" not in move_name:
                move_lines.append(move_name)
    return move_lines

# Detect all Pokémon names mentioned in the text
def detect_pokemon_names(text):
    found = []
    for name in pokemon_db:
        if name.lower() in text.lower():
            found.append(name)
    return found

# Split the full response into blocks per Pokémon
def split_movesets_by_pokemon(text, pokemon_names):
    sections = {}
    current_name = None
    buffer = []

    lines = text.splitlines()
    for line in lines:
        match = next((name for name in pokemon_names if name.lower() in line.lower()), None)
        if match:
            if current_name and buffer:
                sections[current_name] = "\n".join(buffer)
                buffer = []
            current_name = match
        elif current_name:
            buffer.append(line)

    if current_name and buffer:
        sections[current_name] = "\n".join(buffer)

    return sections

# Validate extracted moves against known legal ones
def validate_moves(pokemon_name, moves):
    legal = set(pokemon_db.get(pokemon_name, {}).get("legal_moves", []))
    return [move for move in moves if move not in legal]

# Main chatbot loop
url = "http://127.0.0.1:11434/api/chat"
model_name = "pokemon-assistant-v3"
messages = []

print("Pokémon Assistant LLM (type 'exit' to quit)")
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})
    payload = {"model": model_name, "messages": messages}

    response = requests.post(url, json=payload, stream=True)

    if response.status_code == 200:
        print("Assistant:", end=" ", flush=True)
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

        # Post-generation validation
        pokemon_names = detect_pokemon_names(user_input + " " + full_response)
        if pokemon_names:
            sections = split_movesets_by_pokemon(full_response, pokemon_names)
            for name, section in sections.items():
                moves = extract_moves(section)
                illegal = validate_moves(name, moves)
                if illegal:
                    print(f"\n\u26a0\ufe0f Warning: These moves are illegal for {name}: {', '.join(illegal)}")
                else:
                    print(f"\n\u2705 All moves for {name} are legal.")

        messages.append({"role": "assistant", "content": full_response})
    else:
        print(f"\u274c Error: {response.status_code}")
        print(response.text)
