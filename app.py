import streamlit as st
import requests

# Streamlit Title
st.title("Multilingual Conversational Chatbot")

# Language options
languages = ["English", "Tamil", "French", "Japanese"]
language = st.selectbox("Select your language:", languages, index=0)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}

# Maintain conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history (User & Bot)
for role, content in st.session_state.history:
    if role == "user":
        st.markdown(f"**You ({language}):** {content}")
    else:
        st.markdown(f"**Bot ({language}):** {content}")

# User input field
user_input = st.text_input(f"Type your message in {language}:")

# Send button
if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.history.append(("user", user_input))

    # Prepare prompt: Ask model to always reply in the selected language
    conversation = "\n".join(
        [f"User: {msg}" if role == "user" else f"Assistant: {msg}" for role, msg in st.session_state.history]
    )
    prompt = f"Respond only in {language}. Be helpful and conversational.\n{conversation}\nAssistant:"

    # Call Hugging Face Inference API
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            # Extract text correctly
            output_text = response.json()[0]["generated_text"]
            output_text = output_text.split("Assistant:")[-1].strip()
        except Exception:
            output_text = "Sorry, I couldn't understand the response."
    else:
        output_text = "Error: Could not get a response from the AI model."

    # Add bot response to history and refresh
    st.session_state.history.append(("bot", output_text))
    st.experimental_rerun()
