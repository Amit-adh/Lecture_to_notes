# import ollama

# # Initialize the Ollama client
# client = ollama.Client()

# # Define the model and the input prompt
# model = "llama3.2:3b"  # Replace with your model name
# prompt = "What is Python?"

# # Send the query to the model
# response = client.generate(model=model, prompt=prompt)

# # Print the response from the model
# print("Response from Ollama:")
# print(response.response)

import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b",
        "prompt": "What is the capital of Nepal?",
        "stream": False
    }
)

# Print the full JSON so we can see what's actually returned
print(response.json()['response'])
