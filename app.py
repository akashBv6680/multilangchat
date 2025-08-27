import streamlit as st
import requests

# App Title
st.title("Multilingual Conversational Chatbot")

# Language Selection
languages = ["English", "Tamil", "French", "Japanese"]
language = st.selectbox("Select your language:", languages, index=0)

# Hugging Face API Endpoint (fast conversational model)
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}

# Session State for Chat History
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

# Send Button
if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.history.append(("user", user_input))

    # Prepare prompt (context-aware)
    conversation = "\n".join(
        [f"User: {msg}" if role == "user" else f"Bot: {msg}" for role, msg in st.session_state.history]
    )
    prompt = f"Reply in {language}. Be conversational, natural, and helpful.\n{conversation}\nBot:"

    # API Request to Hugging Face
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code == 200:
        try:
            output_text = response.json().get("generated_text", "").strip()
            if not output_text:
                output_text = "I'm not sure how to respond."
        except Exception:
            output_text = "Error parsing AI response."
    else:
        output_text = f"Error: {response.status_code} - Could not get a response from the AI model."

    # Save bot reply to history
    st.session_state.history.append(("bot", output_text))
