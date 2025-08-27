import streamlit as st
import requests

# App Title
st.title("Multilingual Conversational Chatbot")

# Language options
languages = ["English", "Tamil", "French", "Japanese"]
language = st.selectbox("Select your language:", languages, index=0)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for role, content in st.session_state.history:
    if role == "user":
        st.markdown(f"**You ({language}):** {content}")
    else:
        st.markdown(f"**Bot ({language}):** {content}")

# User input
user_input = st.text_input(f"Type your message in {language}:")

# Send Button Logic
if st.button("Send") and user_input:
    # Add user input to chat history
    st.session_state.history.append(("user", user_input))

    # Prepare conversation prompt
    conversation = "\n".join(
        [f"User: {msg}" if role == "user" else f"Assistant: {msg}" for role, msg in st.session_state.history]
    )
    prompt = f"Respond in {language}. Be conversational, helpful, and context-aware.\n{conversation}\nAssistant:"

    # API call to Hugging Face
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            json_response = response.json()
            if isinstance(json_response, list) and "generated_text" in json_response[0]:
                output_text = json_response[0]["generated_text"].split("Assistant:")[-1].strip()
            else:
                output_text = "Unexpected response format."
        except Exception:
            output_text = "Error parsing AI response."
    else:
        output_text = "Error: Could not get a response from the AI model."

    # Append bot response to chat history
    st.session_state.history.append(("bot", output_text))

    # Force re-render (new messages show instantly without rerun)
    st.session_state.user_input = ""
