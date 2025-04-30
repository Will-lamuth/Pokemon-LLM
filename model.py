import requests
import json
import pandas as pd
import torch
from sklearn.metrics import precision_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments (SSH)
import matplotlib.pyplot as plt

# Set up the base URL for the local Ollama API
url = "http://127.0.0.1:11434/api/chat"

# Load the Pokémon data from the CSV file
pokemon_df = pd.read_csv('Pokemon.csv')

# Clean up the data: Strip spaces from column names and data
pokemon_df.columns = pokemon_df.columns.str.strip()  # Strip column names of extra spaces
pokemon_df['Name'] = pokemon_df['Name'].str.strip()  # Strip Pokémon names of extra spaces
pokemon_df['Type 1'] = pokemon_df['Type 1'].str.strip().str.lower()  # Normalize Type 1
pokemon_df['Type 2'] = pokemon_df['Type 2'].fillna('').str.strip().str.lower()  # Normalize Type 2 and handle NaN

# Function to send requests and handle streaming responses
def get_stream_response(payload):
    try:
        # Send the HTTP POST request with streaming enabled
        response = requests.post(url, json=payload, stream=True)

        # Check the response status
        if response.status_code == 200:
            print("Streaming response from Ollama:")
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Ignore empty lines
                    try:
                        # Parse each line as a JSON object
                        json_data = json.loads(line)
                        # Extract and print the assistant's message content
                        if "message" in json_data and "content" in json_data["message"]:
                            print(json_data["message"]["content"], end="")
                    except json.JSONDecodeError:
                        print(f"\nFailed to parse line: {line}")
            print()  # Ensure the final output ends with a newline
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Function to build a team based on different Pokémon-related requests
def build_pokemon_team(query):
    payload = {
        "model": "llama3-8b-q4",  # Replace with the model name you're using
        "messages": [{"role": "user", "content": query}]
    }
    get_stream_response(payload)

# Function to process the Pokémon data and build a team
def process_pokemon_data():
    # Show the Pokémon data to the user and prompt for a Pokémon to base the team around
    print(pokemon_df[['Name']])  # Display the names of available Pokémon
    selected_pokemon_name = input("Which Pokémon would you like to base your team around? ")

    # Find the selected Pokémon in the data
    selected_pokemon = pokemon_df[pokemon_df['Name'].str.lower() == selected_pokemon_name.lower()]

    if selected_pokemon.empty:
        print(f"Sorry, {selected_pokemon_name} is not in the data.")
        return

    # Formulate the query to build a team based on the selected Pokémon, asking for Smogon-based recommendations
    team_query = f"""
    Build me a competitive Smogon-style team around {selected_pokemon_name}. 
    Please suggest the best EV spreads and movesets for each Pokémon, based on the best competitive teams used in Smogon for {selected_pokemon_name}.
    Make sure the team includes **6 Pokémon** to form a complete team, covering all necessary roles like offensive, defensive, and special attackers, and includes a balance of types (such as Steel/Psychic for Metagross).
    """
    # Build the team and get response
    build_pokemon_team(team_query)

    # Example: Mock predictions and true labels for evaluation
    true_labels = [1, 0, 1, 0, 1, 1]  # Example true labels (ground truth)
    predicted_labels = [1, 0, 1, 0, 0, 1]  # Example predicted labels (from the model)

    # Calculate Precision and F1 Score
    precision = precision_score(true_labels, predicted_labels, average='weighted')  # 'micro', 'macro', or 'weighted'
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Create a table for Precision and F1 Score
    metrics_data = {
        'Metric': ['Precision', 'F1 Score'],
        'Score': [precision, f1]
    }

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table
    print(metrics_df)

    # Visualize Precision and F1 Score using a bar chart
    metrics = ['Precision', 'F1 Score']
    scores = [precision, f1]

    # Plotting the bar chart
    plt.bar(metrics, scores, color=['blue', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance')
    plt.ylim(0, 1)  # Score range from 0 to 1

    # Save the chart to a file (e.g., PNG)
    plt.savefig('performance_chart.png')

    # Optionally, display the chart (if plt.show() works in your environment)
    # plt.show()

    print("Performance chart saved as 'performance_chart.png'.")

# Example usage
if __name__ == "__main__":
    # Process the CSV file to extract Pokémon data and build a team based on user input
    process_pokemon_data()
