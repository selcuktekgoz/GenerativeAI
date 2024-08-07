import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatAnthropic
from langchain_community.chat_models import ChatCohere

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("google_apikey")
anthropic_key = os.getenv("anthropic_apikey")


def ask_model(model_name, prompt, temperature, max_tokens=None):
    if model_name == "gpt-3.5-turbo-0125":
        llm = ChatOpenAI(
            api_key=openai_key,
            temperature=temperature,
            max_tokens=max_tokens,
            model="gpt-3.5-turbo-0125",
        )
    elif model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(
            google_api_key=google_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            model="gemini-pro",
        )
    elif model_name == "claude-2.1":
        llm = ChatAnthropic(
            anthropic_api_key=anthropic_key,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name="claude-2.1",
        )
    else:
        print(
            "Hatalı model ismi. Kullanılan modeller: [gpt-3.5-turbo-0125,gemini-pro,claude-2.1]"
        )

    response = llm.invoke(prompt)

    return response.content


############ HF açık kaynak cosmos modelini kullanalım ############

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import pipeline


def ask_cosmos(prompt, temperature, max_tokens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = 0 if torch.cuda.is_available() else -1

    model = GPT2LMHeadModel.from_pretrained(
        "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
    )

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        device=device_id,
        max_new_tokens=max_tokens,
    )

    def get_model_response(instruction):
        instruction_prompt = f"### Kullanıcı:\n{instruction}\n### Asistan:\n"
        result = text_generator(instruction_prompt)
        generated_response = result[0]["generated_text"]
        return generated_response[len(instruction_prompt) :]

    response = get_model_response(prompt)

    return response
