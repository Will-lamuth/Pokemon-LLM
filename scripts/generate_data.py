import pandas as pd
import json
import random

# Load CSV
df = pd.read_csv("Pokemon.csv")

# Filter out alternate forms like Mega and Primal
df = df[~df['Name'].str.contains("Mega|Primal", na=False)]
df = df.drop_duplicates(subset=["Name"])

# Role inference based on stat spread
def infer_role(row):
    roles = []
    if row['Attack'] >= 100:
        roles.append("physical attacker")
    if row['Sp. Atk'] >= 100:
        roles.append("special attacker")
    if row['Defense'] >= 100 or row['Sp. Def'] >= 100:
        roles.append("defensive wall")
    if row['Speed'] >= 100:
        roles.append("fast sweeper")
    if row['HP'] >= 90:
        roles.append("bulky support")
    if not roles:
        roles.append("situational pick")
    return ", ".join(roles)

# Output list
examples = []

# Generate entries
for _, row in df.iterrows():
    name = row['Name']
    type1 = row['Type 1']
    type2 = row['Type 2'] if pd.notna(row['Type 2']) else None
    types = f"{type1}/{type2}" if type2 else type1
    total = row['Total']
    role = infer_role(row)
    legendary = "a Legendary Pokémon" if row['Legendary'] else "not a Legendary Pokémon"

    # Base stat sentence
    stat_line = (
        f"{name} is a {types}-type Pokémon with {row['HP']} HP, {row['Attack']} Attack, "
        f"{row['Defense']} Defense, {row['Sp. Atk']} Special Attack, "
        f"{row['Sp. Def']} Special Defense, and {row['Speed']} Speed."
    )

    # 1. Battle role description
    examples.append({
        "instruction": f"Describe the battle role of {name} based on its stats.",
        "input": "",
        "output": f"{stat_line} It is {legendary} and plays the role of a {role} in competitive battles."
    })

    # 2. Team-building suggestion
    examples.append({
        "instruction": f"Build a competitive team around {name}.",
        "input": "",
        "output": f"{name} serves as a {role}. A good team should support its weaknesses. Try adding defensive Pokémon like Ferrothorn, "
                  f"offensive partners like Hydreigon, and hazard control such as Rotom-Wash. This helps {name} thrive in battle."
    })

    # 3. Partner recommendation
    examples.append({
        "instruction": f"Suggest a good partner for {name} on a battle team.",
        "input": "",
        "output": f"A solid partner for {name} is something that complements its {types}-typing and {role} role. "
                  f"Examples include bulky pivots like Corviknight or clerics like Clefable depending on your strategy."
    })

    # 4. Threat advice
    examples.append({
        "instruction": f"What threats should I watch out for when using {name}?",
        "input": "",
        "output": f"When using {name}, beware of Pokémon that exploit its weaknesses. Its typing leaves it vulnerable to common threats. "
                  f"Pack team members that can check those threats or pivot into safety when needed."
    })

# Save to JSONL
with open("pokemon_finetune_full.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"✅ Finished! Generated {len(examples)} entries and saved to 'pokemon_finetune_full.jsonl'")