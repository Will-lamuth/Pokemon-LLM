import requests
import json

# Define base URL and model
url = "http://127.0.0.1:11434/api/chat"
model_name = "pokemon-assistant-v2"  # Change to your installed model

# Start conversation history
messages = []

print("Pokémon Assistant LLM (type 'exit' to quit)")
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    # Append user message to chat history
    messages.append({"role": "user", "content": user_input})

    # Send request to Ollama
    payload = {
        "model": model_name,
        "messages": messages
    }

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
        print()  # End the assistant's turn
        # Add assistant reply to chat history
        messages.append({"role": "assistant", "content": full_response})
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)