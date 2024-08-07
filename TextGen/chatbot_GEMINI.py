from os import getenv
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai


load_dotenv()

my_key = getenv("google_apikey")

genai.configure(api_key=my_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

client = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    system_instruction="Sen Büyük Dil Modelleri konusunda bir uzmansın.",
)


def generate_response(prompt: str) -> str:
    """
    Generates a response from the specified model based on the given prompt.

    Parameters:
    prompt (str): The input prompt to generate a response for.

    Returns:
    str: The generated response text from the model.
    """

    chat = client.start_chat(history=[])

    response = chat.send_message(
        content=prompt,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=512),
    )

    return response.text


### Streamlit Interface ###
st.header("TextGen Chat Bot [Gemini Pro]")
st.divider()

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınız"):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_response(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
