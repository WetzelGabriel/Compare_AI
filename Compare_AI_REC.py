import streamlit as st
import pyaudio
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import whisper
from datetime import datetime
import LIB_AudioLib as myALib
import nest_asyncio

def myCompare_REC_App():
    # Apply nest_asyncio if needed
    nest_asyncio.apply()

    # Load the Whisper model
    @st.cache_resource
    def load_whisper_model():
        return whisper.load_model("medium", device="cpu")

    model = load_whisper_model()

    # Get audio output devices
    output_devices, default_device_index = myALib.get_audio_output_devices()
    device_options = [f"{device['name']} (Index: {device['index']})" for device in output_devices]

    # Determine the default index for the radio button
    if default_device_index is not None:
        default_index = next((i for i, d in enumerate(output_devices) if d['index'] == default_device_index), 0)
    else:
        default_index = 0

    # Device selection
    selected_device = st.radio("Select Audio Output Device", device_options, index=default_index)
    selected_device_index = output_devices[device_options.index(selected_device)]['index']

    # Button to start capturing speech
    if st.button("Start Capturing Speech"):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        text_filename = f"text_{timestamp}.txt"
        speech_length = 1

        with st.spinner('Capturing speech...'):
            while speech_length > 0.5:
                # Generate a sound on the selected device for 1 second
                myALib.generate_sound_on_device(device_index=selected_device_index, duration=1)

                # Capture question from the microphone and save it to "input.wav"
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                speech_length = myALib.capture_speech_from_microphone(duration=0, output_filename=f"input_{timestamp}.wav")
                st.write(f"Speech length: {speech_length:.2f} seconds")

                if speech_length > 0.5:
                    result = model.transcribe(f"input_{timestamp}.wav")

                    # Save the transcription to a text file
                    with open(text_filename, "a") as text_file:
                        text_file.write(f"{result['text']}\n\n")

                    # Display the transcription
                    st.write("Transcription:")
                    st.write(result['text'])

        st.success("Speech captured and transcribed successfully!")

        # Option to download the transcription
        with open(text_filename, "r") as file:
            st.download_button(
                label="Download Transcription",
                data=file,
                file_name=text_filename,
                mime='text/plain'
            )

    # Clean up the model when done
    del model

if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    # Streamlit app title
    st.title("Audio Transcription with Whisper")
    myCompare_REC_App()