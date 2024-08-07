from replicate import run
from dotenv import load_dotenv
import streamlit as st

path = "GenerativeAI/TextGen/.env"
load_dotenv(dotenv_path=path, verbose=True)

# If it doesn't work, go to terminal and paste the following.
# export REPLICATE_API_TOKEN=<paste-your-token-here>

system_prompt = "Sen Büyük Dil Modelleri konusunda bir uzmansın."

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "system",
            "message": system_prompt,
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

    response = run(
        "meta/llama-2-70b",
        input={
            "temperature": 0.3,
            "max_new_tokens": 512,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "debug": False,
        },
    )

    return "".join(response)


### Streamlit Interface ###
st.header("TextGen Chat Bot [LLAMA2]")
st.divider()

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajınız"):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_prompt(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
