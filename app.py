# app.py
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

st.title("Multilingual Chatbot")

language = st.selectbox("Choose language:", ["English", "Tamil", "French", "Japanese"])
user_input = st.text_input(f"You ({language}):")

if 'pipeline_obj' not in st.session_state:
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(
        model_name, 
        use_auth_token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    model = MBartForConditionalGeneration.from_pretrained(
        model_name, 
        use_auth_token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    st.session_state.pipeline_obj = (model, tokenizer)

if st.button("Send") and user_input:
    lang_map = {
        "English": "en_XX",
        "Tamil": "ta_IN",
        "French": "fr_XX",
        "Japanese": "ja_XX"
    }
    tgt_lang = lang_map[language]

    model, tokenizer = st.session_state.pipeline_obj
    tokenizer.src_lang = "en_XX"  # Assuming input is English for now

    encoded = tokenizer(user_input, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=200
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    st.text_area("Bot:", value=result, height=150)
