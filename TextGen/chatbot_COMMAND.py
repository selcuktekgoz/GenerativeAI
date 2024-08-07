from os import getenv
from dotenv import load_dotenv
import streamlit as st
from cohere import Client


load_dotenv()

my_key = getenv("cohere_apikey")

client = Client(api_key=my_key)


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "CHATBOT",
            "message": "Sen Büyük Dil Modelleri konusunda bir uzmansın.",
        }
    )


def generate_prompt(prompt: str) -> str:
    """
    Generates a response from the specified model based on the given prompt.

    Parameters:
    prompt (str): The input prompt to generate a response for.

    Returns:
    str: The generated response text from the model.
    """

    response = client.chat(
        model="command",
        temperature=0,
        max_tokens=512,
        chat_history=st.session_state.messages,
        message=prompt,
    )

    return response.text


### Streamlit Interface ###
st.header("TextGen Chat Bot [Command]")
st.divider()

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if prompt := st.chat_input("Mesajınız"):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_prompt(prompt)
    st.session_state.messages.append({"role": "USER", "message": prompt})
    st.session_state.messages.append({"role": "CHATBOT", "message": response})

    with st.chat_message("assistant"):
        st.markdown(response)
