from os import getenv
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv()

my_key = getenv("OPENAI_API_KEY")

client = OpenAI(api_key=my_key)


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "system", "content": "Sen Büyük Dil Modelleri konusunda bir uzmansın."}
    )


def generate_response(prompt: str) -> str:
    """
    Generates a response from the specified model based on the given prompt.

    Parameters:
    prompt (str): The input prompt to generate a response for.

    Returns:
    str: The generated response text from the model.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=1024,
        messages=st.session_state.messages,
    )

    return response.choices[0].message.content


### Streamlit Interface ###
st.header("TextGen Chat Bot [Gpt 3.5 Turbo]")
st.divider()


for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınız"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
