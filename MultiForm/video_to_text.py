from dotenv import load_dotenv
from os import getenv, path
from tempfile import mkdtemp
import streamlit as st
import assemblyai as aai

load_dotenv()

my_key = getenv("assemblyai_apikey")

aai.settings.api_key = my_key

config = aai.TranscriptionConfig(language_code="tr")


def video_to_srt(video_file_name):
    transcript = aai.Transcriber(config=config).transcribe(video_file_name)
    subtitles = transcript.export_subtitles_srt()
    with open("assets/content.srt", "w") as f:
        f.write(subtitles)
    return subtitles


####### Streamlit Interface #######

st.set_page_config(
    page_title="Video Content Detection", page_icon=":robot_face:", layout="centered"
)

st.header("Video to Text [AssemblyAI]")
st.divider()


selected_file = st.file_uploader(
    label="Select a video", type=["mp4", "mpeg4"], key="selected_file"
)


if selected_file:
    temp_dir = mkdtemp()
    path = path.join(temp_dir, selected_file.name)
    with open(path, "wb") as f:
        f.write(selected_file.getvalue())
    audio_file = open(path, "rb")
    audio_bytes = audio_file.read()
    st.video(audio_bytes, "video/mp4")


run_button = st.button("Run", key="run_button")

if run_button:
    try:
        content = video_to_srt(path)
        st.divider()
        st.success("Proccess finished")
        st.info(content)
    except Exception as e:
        st.error(e)
