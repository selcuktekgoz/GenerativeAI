from os import getenv
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

my_key = getenv("anthropic_apikey")

client = Anthropic(api_key=my_key)


if "messages" not in st.session_state:
    st.session_state.messages = []


def generate_prompt(prompt: str) -> str:
    """
    Generates a response from the specified model based on the given prompt.

    Parameters:
    prompt (str): The input prompt to generate a response for.

    Returns:
    str: The generated response text from the model.
    """

    response = client.messages.create(
        model="claude-2.1",
        system="Sen Büyük Dil Modelleri konusunda bir uzmansın.",
        temperature=0,
        max_tokens=512,
        messages=st.session_state.messages,
    )

    return response.content[0].text


### Streamlit Interface ###
st.header("TextGen Chat Bot [Claude 2.1]")
st.divider()

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınız"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_prompt(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
