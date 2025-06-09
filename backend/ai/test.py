from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Use MPS if available, else fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
model = model.to(device)

input_text = ""
while input_text != "exit":
    input_text = input("Enter text to translate (or 'exit' to quit): ")
    if input_text.lower() == 'exit':
        print("Exiting...")
        exit()
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate output
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
