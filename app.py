import streamlit as st
import requests

# Streamlit App Title
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

# Display chat history
for msg in st.session_state.history:
    role, content = msg
    if role == "user":
        st.markdown(f"**You ({language}):** {content}")
    else:
        st.markdown(f"**Bot ({language}):** {content}")

# User input box
user_input = st.text_input(f"Type your message in {language}:")

# Send message
if st.button("Send") and user_input:
    # Add user input to history
    st.session_state.history.append(("user", user_input))

    # Build conversation prompt
    conversation = "\n".join([f"{'User' if role=='user' else 'Assistant'}: {content}" for role, content in st.session_state.history])
    prompt = f"Reply in {language}. Continue the conversation:\n{conversation}\nAssistant:"

    # Call Hugging Face API
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        output_text = response.json()[0]["generated_text"].split("Assistant:")[-1].strip()
    else:
        output_text = "Error: Could not get a response from the AI model."

    # Add bot reply to h
