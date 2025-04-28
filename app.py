import os
import base64
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize floating components
float_init()

# Title
st.title("OpenAI Conversational Chatbot 🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How may I assist you today?"}
    ]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Footer container for mic input
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

# Speech-to-Text Function
def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en"
        )
    return transcript

# Get ChatGPT Response
def get_answer(messages):
    system_prompt = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=system_prompt + messages
    )
    return response.choices[0].message.content

# Text-to-Speech Function
def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=input_text
    )
    file_path = "temp_audio_play.mp3"
    with open(file_path, "wb") as f:
        response.stream_to_file(file_path)
    return file_path

# Autoplay Audio in Streamlit
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# If audio recorded, transcribe it
if audio_bytes:
    with st.spinner("Transcribing..."):
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        transcript = speech_to_text(audio_path)
        os.remove(audio_path)

    if transcript:
        st.session_state.messages.append({"role": "user", "content": transcript})
        with st.chat_message("user"):
            st.write(transcript)

# Generate assistant reply
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking 🤔..."):
            answer = get_answer(st.session_state.messages)
        with st.spinner("Generating audio response..."):
            audio_file = text_to_speech(answer)
            autoplay_audio(audio_file)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        os.remove(audio_file)

# Float footer to bottom
footer_container.float("bottom: 0rem;")
