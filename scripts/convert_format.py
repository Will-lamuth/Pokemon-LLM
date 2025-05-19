import json

def convert_moveset_jsonl_to_dict(jsonl_path, output_path):
    pokemon_db = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            instruction = entry["instruction"]
            output = entry["output"]

            # Extract Pokémon name from instruction
            # Assumes format like "Which moves can Bulbasaur learn?"
            name = instruction.replace("Which moves can ", "").replace(" learn?", "").strip()

            moves = [move.strip() for move in output.split(",") if move.strip()]
            pokemon_db[name] = {"legal_moves": moves}

    # Save to JSON for use in filtering
    with open(output_path, "w") as out:
        json.dump(pokemon_db, out, indent=2)

# Usage
convert_moveset_jsonl_to_dict("../data/unsloth_formatted_dataset.jsonl", "pokemon_moveset_db.json")
