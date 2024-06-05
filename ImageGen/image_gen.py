from openai import OpenAI
from os import getenv, path
from tempfile import mkdtemp
from dotenv import load_dotenv
import streamlit as st
from io import BytesIO
from base64 import b64decode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

openai_key = getenv("OPENAI_API_KEY")
stabilityai_key = getenv("stabilityai_apikey")

client = OpenAI(api_key=openai_key)


retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


def generate_image(prompt):

    response = client.images.generate(
        model="dall-e-3",
        size="1024x1024",
        quality="hd",
        n=1,
        response_format="url",  # or b64_json.
        prompt=prompt,
    )

    image_url = response.data[0].url
    improved_prompt = response.data[0].revised_prompt

    image_response = requests.get(image_url)
    image_bytes = BytesIO(image_response.content)

    return image_bytes, improved_prompt


def generate_image_variation(image_url):

    try:
        response = client.images.create_variation(
            image=open(image_url, "rb"), size="512x512", n=1, response_format="url"
        )

        image_url = response.data[0].url

        # image_response = requests.get(image_url)
        image_response = http.get(image_url, timeout=30)
        image_response.raise_for_status()
        image_bytes = BytesIO(image_response.content)

        return image_bytes

    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred: {e}")
        return None


def generate_image_with_sd(prompt):

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {stabilityai_key}",
    }

    json = {
        "text_prompts": [
            {"text": prompt, "weight": 0.7},
            {"text": "bad photo, bluury", "weight": -1},  # negative prompt
        ],
        "steps": 10,  # Number of diffusion steps to run.
        "width": 1024,
        "height": 1024,
        "cfg_scale": 7,  # How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)
        "samples": 1,  # Number of images to generate
        "style_preset": "3d-model",  # Pass in a style preset to guide the image model towards a particular style. This list of style presets is subject to change.
    }

    response = requests.post(url, headers=headers, json=json)

    data = response.json()

    return data


with st.sidebar:
    add_radio = st.radio(
        "Choose an image operation",
        ("Image Generate", "Image Variation", "Stable Diffusion"),
    )

with st.container():
    if add_radio == "Image Generate":
        st.subheader("Creating Visuals with DALL-E 3")
        st.divider()
        generate_prompt = st.text_input(
            "Describe the image you want to create :", key="generate_prompt"
        )
        generate_btn = st.button("Draw", key="generate_btn")

        if generate_btn:
            image_data, improved_prompt = generate_image(generate_prompt)

            st.image(image=image_data)
            st.divider()
            st.write("Improved prompt :")
            st.caption(improved_prompt)

    if add_radio == "Image Variation":
        st.subheader("Creating Variations with DALL-E 3")
        st.divider()
        selected_file = st.file_uploader("Choose a png file", type=["png"])

        if selected_file:
            temp_dir = mkdtemp()
            path = path.join(temp_dir, selected_file.name)
            with open(path, "wb") as f:
                f.write(selected_file.getvalue())
            image_file = open(path, "rb")
            image_bytes = image_file.read()
            st.image(image_bytes)

        variation_btn = st.button("Draw", key="variation_btn")

        if variation_btn:
            image_data = generate_image_variation(path)

            st.image(image=image_data)

    if add_radio == "Stable Diffusion":
        st.subheader("Creating Image with Stable Diffusion XL")
        st.divider()

        sd_prompt = st.text_input(
            "Describe the image you want to create :", key="sd_prompt"
        )
        sd_generate_btn = st.button("Draw", key="sd_generate_btn")

        if sd_generate_btn:
            data = generate_image_with_sd(sd_prompt)

            for image in data["artifacts"]:
                image_bytes = b64decode(image["base64"])
                st.image(image=image_bytes)
