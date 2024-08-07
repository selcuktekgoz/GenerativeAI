import streamlit as st
import helper
import time

st.set_page_config(page_title="LangChain: Model Karşılaştırma", layout="wide")
st.title("LangChain: Model Karşılaştırma")
st.divider()

col_prompt, col_settings = st.columns([3, 2])

with col_prompt:
    prompt = st.text_input(label="Sorunuzu giriniz:")
    st.divider()
    submit_btn = st.button("Sor")

with col_settings:
    temperature = st.slider(
        label="Temperature", min_value=0.0, max_value=1.0, value=0.7
    )
    max_tokens = st.slider(
        label="Maximum Tokens", min_value=128, max_value=1024, value=256, step=64
    )

st.divider()


def get_model_response(model_name):
    start_time = time.perf_counter()

    if model_name == "cosmos":
        response = helper.ask_cosmos(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens
        )
    else:
        response = helper.ask_model(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    st.write(response)
    st.caption(f"| :hourglass: {round(elapsed_time)} saniye")


col_gpt, col_gemini, col_cosmos = st.columns(3)

with col_gpt:
    if submit_btn:
        with st.spinner("GPT Yanıtlıyor..."):
            st.success("GPT-3.5 Turbo")
            get_model_response(model_name="gpt-3.5-turbo-0125")


with col_gemini:
    if submit_btn:
        with st.spinner("Gemini Yanıtlıyor..."):
            st.info("Gemini Pro")
            get_model_response(model_name="gemini-pro")


with col_cosmos:
    if submit_btn:
        with st.spinner("Cosmos Yanıtlıyor..."):
            st.error("Cosmos Gpt2")
            get_model_response(model_name="cosmos")
