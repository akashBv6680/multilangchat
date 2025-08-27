# app.py
import streamlit as st
from transformers import pipeline

st.title("Multilingual Chatbot")

# Select language
language = st.selectbox("Choose language:", ["English", "Tamil", "French", "Japanese"])
user_input = st.text_input(f"You ({language}):")

# Setup pipeline
if 'pipeline_obj' not in st.session_state:
    st.session_state.pipeline_obj = pipeline(
        "text2text-generation",
        model="facebook/mbart-large-50-many-to-many-mmt",
        tokenizer="facebook/mbart-large-50-many-to-many-mmt",
        use_auth_token=st.secrets["HUGGINGFACE_TOKEN"]
    )

if st.button("Send"):
    # Map language to mbart code (e.g., 'en_XX', 'ta_IN', 'fr_XX', 'ja_XX')
    lang_map = {
        "English": "en_XX",
        "Tamil": "ta_IN",
        "French": "fr_XX",
        "Japanese": "ja_XX"
    }
    tgt_lang = lang_map[language]
    prompt = f">>{tgt_lang}<< {user_input}"
    res = st.session_state.pipeline_obj(prompt, max_length=200)
    st.text_area("Bot:", value=res[0]['generated_text'], height=150)
