import threading
import time

import streamlit as st
from dotenv import load_dotenv

from helper import VoiceRecorder, Transcriptor, Painter

load_dotenv()

st.set_page_config(
    page_title="Draw With Voice", layout="wide", page_icon="./assets/icon.png"
)


def main():

    recorder = VoiceRecorder()
    transcriptor = Transcriptor()
    painter = Painter()

    if "record_on" not in st.session_state:
        st.session_state.record_on = threading.Event()
        st.session_state.recording_status = "Ready!"
        st.session_state.recording_completed = False
        st.session_state.latest_image = ""
        st.session_state.messages = []

    st.image(image="./assets/top_image.png", use_column_width=True)

    with st.sidebar:
        st.sidebar.image("./assets/icon.png", use_column_width=True)
        st.header("Record your voice")
        st.divider()
        status_message = st.success(st.session_state.recording_status)
        st.divider()

        subcol_left, subcol_right = st.columns([1, 2])

        with subcol_left:
            start_btn = st.button(
                label="Start",
                on_click=lambda: start_recording(recorder),
                disabled=st.session_state.record_on.is_set(),
            )
            stop_btn = st.button(
                label="Stop",
                on_click=lambda: stop_recording(),
                disabled=not st.session_state.record_on.is_set(),
            )
        with subcol_right:
            recorded_audio = st.empty()

            if st.session_state.recording_completed:
                with st.spinner("Please wait.."):
                    time.sleep(1)
                    recorded_audio.audio(data="voice.wav")

        st.divider()
        latest_image_use = st.checkbox(label="Update the last image")

    with st.container():
        st.subheader("Outputs")
        st.divider()

        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message(name=message["role"], avatar="./assets/ai.png"):
                    st.warning("Output:")
                    st.image(image=message["content"], width=300)
            elif message["role"] == "user":
                with st.chat_message(name=message["role"], avatar="./assets/user.png"):
                    st.success(message["content"])

        if stop_btn:
            with st.chat_message(name="user", avatar="./assets/user.png"):
                with st.spinner("Please wait.."):
                    voice_prompt = transcriptor.transcribe_audio(
                        audio_file_name="voice.wav"
                    )
                st.success(voice_prompt)

            st.session_state.messages.append({"role": "user", "content": voice_prompt})

            with st.chat_message(name="assistant", avatar="./assets/ai.png"):
                st.warning("Output:")
                with st.spinner("Please wait.."):
                    if latest_image_use:
                        image_file_name = painter.update_image(
                            image_path=st.session_state.latest_image,
                            prompt=voice_prompt,
                        )
                    else:
                        image_file_name = painter.generate_image(prompt=voice_prompt)

                st.image(image=image_file_name, width=300)

                with open(image_file_name, "rb") as file:
                    st.download_button(
                        label="Download",
                        data=file,
                        file_name=image_file_name,
                        mime="image/png",
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": image_file_name}
            )
            st.session_state.latest_image = image_file_name


def start_recording(recorder):
    st.session_state.record_on.set()
    st.session_state.recording_status = "Recording.."
    st.session_state.recording_completed = False

    threading.Thread(
        target=recorder.record,
        args=(st.session_state.record_on, recorder.frames),
    ).start()


def stop_recording():
    st.session_state.record_on.clear()
    st.session_state.recording_status = "Recording completed successfully."
    st.session_state.recording_completed = True


if __name__ == "__main__":
    main()
