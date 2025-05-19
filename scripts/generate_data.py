import requests
import json
import time

POKEAPI_BASE = "https://pokeapi.co/api/v2"

# Generations 1 and 2 are numbered 1 and 2 in PokeAPI
def get_pokemon_species_by_generation(gen):
    url = f"{POKEAPI_BASE}/generation/{gen}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return [entry['name'] for entry in data['pokemon_species']]

def get_legal_moves(pokemon_name):
    url = f"{POKEAPI_BASE}/pokemon/{pokemon_name.lower()}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch data for {pokemon_name}")
        return []
    data = resp.json()
    legal_moves = set()
    for move in data['moves']:
        for detail in move['version_group_details']:
            version_group = detail['version_group']['name']
            move_learn_method = detail['move_learn_method']['name']
            if version_group in ["red-blue", "yellow", "gold-silver", "crystal"]:
                legal_moves.add(move['move']['name'].replace("-", " ").title())
    return sorted(legal_moves)

all_pokemon = {}

gen1 = get_pokemon_species_by_generation(1)
gen2 = get_pokemon_species_by_generation(2)
all_species = sorted(set(gen1 + gen2))

for i, name in enumerate(all_species, 1):
    print(f"[{i}/{len(all_species)}] Fetching: {name}")
    moves = get_legal_moves(name)
    all_pokemon[name.title()] = {
        "generation": 1 if name in gen1 else 2,
        "legal_moves": moves
    }
    time.sleep(0.5)  # Be nice to PokeAPI

with open("pokemon_gen12_moveset.json", "w") as f:
    json.dump(all_pokemon, f, indent=2)

print("✅ Saved as pokemon_gen12_moveset.json")
