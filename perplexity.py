import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def calculate_perplexity(text, model, tokenizer):
    # Tokenize the text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Compute perplexity from the loss
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_burstiness(text):
    # Basic burstiness calculation (frequency of words)
    words = text.split()
    word_frequency = {word: words.count(word) for word in set(words)}
    burstiness = len(word_frequency) / len(words)
    return burstiness

def main():
    st.title("Perplexity and Burstiness Calculator")

    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Get user input
    user_input = st.text_area("Enter text:", "The quick brown fox jumps over the lazy dog.")

    # Calculate perplexity when the user clicks the button
    if st.button("Calculate Metrics"):
        with st.spinner("Calculating metrics..."):
            perplexity = calculate_perplexity(user_input, model, tokenizer)
            burstiness = calculate_burstiness(user_input)

        # Display results
        st.write(f"Perplexity: {perplexity:.2f}")
        st.write(f"Burstiness: {burstiness:.2%}")

if __name__ == "__main__":
    main()