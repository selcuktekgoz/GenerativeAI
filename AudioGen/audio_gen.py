from dotenv import load_dotenv
from os import getenv, path
from tempfile import mkdtemp
import streamlit as st
import assemblyai as aai
from openai import OpenAI


load_dotenv()

aai_key = getenv("assemblyai_apikey")
openai_key = getenv("OPENAI_API_KEY")

aai.settings.api_key = aai_key
config = aai.TranscriptionConfig(language_code="tr")

client = OpenAI(api_key=openai_key)


#################### Streamlit Interface ####################


def speech_to_text(prompt, audio_file, voice_type="shimmer"):

    response = client.audio.speech.create(
        model="tts-1", voice=voice_type, response_format="mp3", input=prompt
    )

    response.stream_to_file(audio_file)

    return "The operation completed successfully."


def transcribe_with_whisper(audio_file_name):

    audio_file = open(audio_file_name, "rb")

    response = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="tr"
    )

    return response.text


def translate_with_whisper(audio_file_name):
    audio_file = open(audio_file_name, "rb")

    response = client.audio.translations.create(model="whisper-1", file=audio_file)

    return response.text


def transcribe_with_aai(audio_file_name):
    transcript = aai.Transcriber(config=config).transcribe(audio_file_name)
    subtitles = transcript.export_subtitles_srt()
    return subtitles


with st.sidebar:
    add_radio = st.radio(
        "Choose an audio operation",
        (
            "TTS",
            "Transcription-1",
            "Translate",
            "Transcription-2",
        ),
    )

with st.container():
    if add_radio == "TTS":
        st.subheader("Text To Speach <OPENAI>")
        st.divider()

        prompt_tts = st.text_input("Enter text :", key="prompt_tts")
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice_type = st.selectbox(label="Choose one:", options=voices, key="voice_tts")
        btn_tts = st.button("Run", key="btn_tts")

        if btn_tts:
            status = speech_to_text(prompt_tts, "audio.mp3", voice_type)
            st.success(status)

            audio_file = open("audio.mp3", "rb")
            audio_bytes = audio_file.read()

            st.audio(data=audio_bytes, format="audio/mp3")

    if add_radio == "Transcription-1":
        st.subheader("Transcription <Whisper>")
        st.divider()

        selected_file = st.file_uploader(
            label="Select an audio", type=["mp3"], key="selected_audio"
        )

        if selected_file:
            temp_dir = mkdtemp()
            path = path.join(temp_dir, selected_file.name)
            with open(path, "wb") as f:
                f.write(selected_file.getvalue())
            audio_file = open(path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, "audio/mp3")

        btn_transcribe = st.button("Run", key="btn_transcribe")

        if btn_transcribe:
            try:
                content = transcribe_with_whisper(path)
                st.divider()
                st.success("Proccess finished")
                st.info(f"TRANSCRIPTION : {content}")
            except Exception as e:
                st.error(e)

    if add_radio == "Translate":
        st.subheader("Translate <Whisper>")
        st.divider()

        selected_file = st.file_uploader(
            label="Select an audio", type=["mp3"], key="selected_audio_translate"
        )

        if selected_file:
            temp_dir = mkdtemp()
            path = path.join(temp_dir, selected_file.name)
            with open(path, "wb") as f:
                f.write(selected_file.getvalue())
            audio_file = open(path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, "audio/mp3")

        btn_translate = st.button("Run", key="btn_translate")

        if btn_translate:
            try:
                content = translate_with_whisper(path)
                st.divider()
                st.success("Proccess finished")
                st.info(f"TRANSLATION : {content}")
            except Exception as e:
                st.error(e)

    if add_radio == "Transcription-2":
        st.subheader("Transcription <ASSEMBLYAI>")
        st.divider()

        selected_file = st.file_uploader(
            label="Select an audio", type=["mp3"], key="selected_audio_aai"
        )

        if selected_file:
            temp_dir = mkdtemp()
            path = path.join(temp_dir, selected_file.name)
            with open(path, "wb") as f:
                f.write(selected_file.getvalue())
            audio_file = open(path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, "audio/mp3")

        btn_transcribe_aai = st.button("Run", key="btn_transcribe_aai")

        if btn_transcribe_aai:
            try:
                content = transcribe_with_aai(path)
                st.divider()
                st.success("Proccess finished")
                st.info(f"TRANSCRIPTION : {content}")
            except Exception as e:
                st.error(e)
