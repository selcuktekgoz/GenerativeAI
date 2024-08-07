import streamlit as st
from dotenv import load_dotenv
from os import getenv, path
from openai import OpenAI
import google.generativeai as genai
import anthropic


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "system", "content": "Sen python programlama dili uzmanısın."}
    )


def history_reset():
    st.session_state.messages = st.session_state.messages[:1]


#################### open ai key ##########################
openai_key = getenv("OPENAI_API_KEY")
client_gpt = OpenAI(api_key=openai_key)


def gpt_response(prompt: str) -> str:

    response = client_gpt.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=512,
        messages=st.session_state.messages,
    )

    return response.choices[0].message.content


#################### gemini key ##########################
gemini_key = getenv("google_apikey")
genai.configure(api_key=gemini_key)
client_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    system_instruction=st.session_state.messages[0]["content"],
)


def gemini_response(prompt: str) -> str:
    chat = client_gemini.start_chat(history=[])
    response = chat.send_message(
        content=prompt,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=512),
    )
    return response.text


#################### anthropic key ##########################
anthropic_key = getenv("anthropic_apikey")
client_anthropic = anthropic.Anthropic(api_key=anthropic_key)


def claude_response(prompt: str) -> str:
    response = client_anthropic.messages.create(
        model="claude-2.1",
        system=st.session_state.messages[0]["content"],
        temperature=0,
        max_tokens=512,
        messages=st.session_state.messages,
    )

    return response.content[0].text


def generate_response(prompt: str) -> str:
    """
    Generates a response from the specified model based on the given prompt.

    Parameters:
    prompt (str): The input prompt to generate a response for.

    Returns:
    str: The generated response text from the model.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.selected_model == "Gpt 3.5 Turbo":
        response = gpt_response(prompt)

    elif st.session_state.selected_model == "Gemini Pro":
        response = gemini_response(prompt)

    else:
        response = claude_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.sidebar.header("Code Generation App")
    st.sidebar.divider()

    models = ["Gpt 3.5 Turbo", "Gemini Pro", "Claude 2.1"]
    selected_model = st.sidebar.selectbox(
        label="Dil Modelini Seçiniz:",
        options=models,
        on_change=history_reset,
        key="selected_model",
    )


if selected_model == "Gpt 3.5 Turbo":
    st.subheader("Code-Gen <Gpt 3.5>")
elif selected_model == "Gemini Pro":
    st.subheader("Code-Gen <Gemini Pro>")
else:
    st.subheader("Code-Gen <Claude>")

st.divider()

if prompt := st.chat_input("Mesajınız"):
    generate_response(prompt)

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.session_state
