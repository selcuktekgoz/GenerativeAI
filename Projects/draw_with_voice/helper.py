import os
import wave
from datetime import datetime
from io import BytesIO

import PIL
import assemblyai as aai
import google.generativeai as genai
import pyaudio  # sudo apt install portaudio19-dev | pip install pyaudio

import requests
from openai import OpenAI


class VoiceRecorder:
    def __init__(
        self, filename="voice.wav", channels=1, rate=44100, frames_per_buffer=1024
    ):
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.frames = []
        self.record_on = None

    def record(self, record_on, frames):
        self.record_on = record_on
        audio = pyaudio.PyAudio()

        voice_stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )

        while self.record_on.is_set():
            data = voice_stream.read(
                self.frames_per_buffer, exception_on_overflow=False
            )
            frames.append(data)

        voice_stream.stop_stream()
        voice_stream.close()
        audio.terminate()

        voice_file = wave.open(self.filename, "wb")
        voice_file.setnchannels(self.channels)
        voice_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        voice_file.setframerate(self.rate)
        voice_file.writeframes(b"".join(frames))
        voice_file.close()


class Transcriptor:
    def __init__(self):
        self.aai_key = os.getenv("assemblyai_apikey")
        aai.settings.api_key = self.aai_key
        self.config = aai.TranscriptionConfig(language_code="tr")

    def transcribe_audio(self, audio_file_name):
        transcript = aai.Transcriber(config=self.config).transcribe(audio_file_name)
        text = transcript.text
        return text


class Painter:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_key)
        self.google_key = os.getenv("google_apikey")
        genai.configure(api_key=self.google_key)
        self.genai_client = genai

    def generate_image(self, prompt):
        response = self.openai_client.images.generate(
            model="dall-e-3",
            size="1024x1024",
            quality="hd",
            n=1,
            response_format="url",
            prompt=prompt,
        )

        image_url = response.data[0].url
        improved_prompt = response.data[0].revised_prompt

        image_response = requests.get(image_url)
        image_bytes = BytesIO(image_response.content)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./img/image_{timestamp}.png"

        if not os.path.exists("./img"):
            os.makedirs("./img")

        with open(filename, "wb") as file:
            file.write(image_bytes.getbuffer())

        return filename

    def update_image(self, image_path, prompt):
        update_prompt = f"""Bu resmi ayrıntılı bir şekilde betimle ve ek yönergeyi dikkate alarak yeniden oluştur. 
        İşte ek yönerge: {prompt}"""

        client = self.genai_client.GenerativeModel(model_name="gemini-pro-vision")

        source_image = PIL.Image.open(image_path)

        response = client.generate_content([update_prompt, source_image])

        response.resolve()

        return self.generate_image(response.text)
