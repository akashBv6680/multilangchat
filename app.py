import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

st.title("Multilingual Chatbot")

# Language mappings for mBART-50
lang_map = {
    "English": "en_XX",
    "Tamil": "ta_IN",
    "French": "fr_XX",
    "Japanese": "ja_XX"
}

# Dropdown for input & output languages
input_lang = st.selectbox("Select Input Language:", list(lang_map.keys()), index=0)
output_lang = st.selectbox("Select Output Language:", list(lang_map.keys()), index=0)

# User input text
user_input = st.text_input(f"You ({input_lang}):")

# Load model & tokenizer once
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

# Generate response when user clicks Send
if st.button("Send") and user_input:
    model, tokenizer = st.session_state.pipeline_obj

    # Set source language
    tokenizer.src_lang = lang_map[input_lang]

    # Tokenize input
    encoded = tokenizer(user_input, return_tensors="pt")

    # Generate text in target language
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[lang_map[output_lang]],
        max_length=200
    )

    # Decode result
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    st.text_area(f"Bot ({output_lang}):", value=result, height=150)
