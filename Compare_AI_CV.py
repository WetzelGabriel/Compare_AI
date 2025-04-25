# to run: 
# streamlit run /home/gabriel/myProject/myCode/xyz.py
import pyaudio
import sounddevice as sd
import numpy as np
import time
import speech_recognition as sr
import whisper
import torch
import os

from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

import LIB_AudioLib as myALib

import asyncio
import nest_asyncio

# Apply nest_asyncio if needed
nest_asyncio.apply()
import nest_asyncio

from datetime import datetime
import shutil

# Streamlit-App
def myCompare_CV_App():
    st.title("Candidate Interview")
    file_path = "/home/gabriel/myProject/myDocs/CV_Gabriel_250302_G.pdf"
    documents = asyncio.run(myALib.upload_file(file_path))

    # GPU-Speicher freigeben
    torch.cuda.empty_cache()
    # Load the Whisper model
    model = whisper.load_model("small", device="cpu")

    output_devices, default_device_index = myALib.get_audio_output_devices()

    # Format the options for the radio button
    options = [f"{device['name']} (Index: {device['index']})" for device in output_devices]

    # Determine the default index for the radio button
    if default_device_index is not None:
        default_index = next((i for i, d in enumerate(output_devices) if d['index'] == default_device_index), 0)
    else:
        default_index = 0

    # Select box with the audio output devices
    selected_device = st.selectbox(
        "Wählen Sie ein Audio-Outputgerät:",
        options=options,
        index=default_index
    )

    # Extract the selected device index
    selected_device_index = output_devices[options.index(selected_device)]['index']

    # Hier können Sie den ausgewählten Audio-Output verwenden
    st.write(f"Ausgewähltes Gerät: {selected_device} (Index: {selected_device_index})")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


    # Container for chat messages
    chat_container = st.container()

    # Initialize the variable i outside the button click event
    if 'i' not in st.session_state:
        st.session_state.i = 2
    if 'my_stop_chat' not in st.session_state:
        st.session_state.my_stop_chat = False

    # Button to start the interview
    if st.button("Start der Interview"):
        st.session_state.chat_history = []  # Clear chat history on start
        st.session_state.stop_chat = False  # Initialize stop flag

        # Example usage: Generate a sound on device for 1 second
        myALib.generate_sound_on_device(device_index=selected_device_index, duration=1)

        question = """
        You are a candidate for a job interview.
        Your resume is the file that is uploaded.
        In this file, you find everything about you,
        that you should know
        Your task is to respond to questions in German
        that the human ressources manager is asking you.
        The human resource manager is sitting in front of you.
        You only respond to the following question: Say hello to him in few nice words.
        """
        
        response = myALib.ask_question(documents, question)
        with chat_container:
            st.chat_message("Candidate").markdown(response)
        st.session_state.chat_history.append({"role": "Candidate", "content": {response}})

        myALib.pico_text_to_speech(response, output_filename="output1.wav")
        # Play the response file "output1.wav" on device
        myALib.play_speech_on_device(file_path="output1.wav", device_index=selected_device_index)
        # Example usage: Generate a sound on device for 1 second
        myALib.generate_sound_on_device(device_index=selected_device_index, duration=1)

    
        while True:
            myALib.generate_sound_on_device(device_index=selected_device_index, duration=1)

            # Capture question from the microphone for 10 seconds and save it to "input.wav"
            myALib.capture_speech_from_microphone(duration=10, output_filename=f"input{st.session_state.i}.wav")
            # Transcribe an audio file
            result = model.transcribe(f"input{st.session_state.i}.wav")
            # Print the transcription
            with chat_container:
                st.chat_message("Interviewer").markdown(result["text"])
            st.session_state.chat_history.append({"role": "Interviewer", "content": {result["text"]}})

            question = f"""
            You are a candidate for a job interview.
            Your resume is the file that is uploaded.
            In this file, you find everything about you,
            that you should know
            Your task is to respond to quimport streamlit as st
    estions in German
            that the human ressources manager is asking you.
            The human resource manager is sitting in front of you.
            You only respond to the following question:
            {result["text"]}
            """

            response = myALib.ask_question(documents, question)
            myALib.pico_text_to_speech(response, output_filename=f"output{st.session_state.i}.wav")
            # Play the response file "output.wav" on device
            with chat_container:
                st.chat_message("Candidate").markdown(response)
            st.session_state.chat_history.append({"role": "Candidate", "content": {response}})
            myALib.play_speech_on_device(file_path=f"output{st.session_state.i}.wav", device_index=selected_device_index)
            # Example usage: Generate a sound on device for 1 second
            
            st.session_state.i = st.session_state.i + 1
            myALib.generate_sound_on_device(device_index=selected_device_index, duration=1)

            # Check if the stop button was pressed
            if st.session_state.my_stop_chat:
                st.session_state.stop_chat = True
                st.write("Interview gestoppt.")
                del model  # Modell löschen
                torch.cuda.empty_cache()  # Speicher freigeben
                break

    # Button to stop the chat
    if st.button("Stop der Interview"):
        st.session_state.my_stop_chat = True
        st.write("Interview gestoppt.")
        
    if st.button("Save the interview"):
        # Aggregate all input and output files into output1.wav
        # Get the current date and time

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Create the masterfile name with date and time
        masterfile = f"interview_{timestamp}.wav"

        # Copy output1.wav to masterfile as the start value
        shutil.copy("output1.wav", masterfile)
        for j in range(2, st.session_state.i+1):
            st.write(f"...{j}...")
            input_file = f"input{j}.wav"
            output_file = f"output{j}.wav"
            if os.path.exists(input_file):
                masterfile = myALib.my_wav_aggregator(masterfile, input_file)
            else:
                st.write(f"File {input_file} does not exist and will be skipped.")
            if os.path.exists(output_file):
                masterfile = myALib.my_wav_aggregator(masterfile, output_file)
            else:
                st.write(f"File {output_file} does not exist and will be skipped.")
        st.write("st.session_state.i: ", st.session_state.i)
        st.write(f"Alle Dateien wurden zu {masterfile} zusammengefasst.")

        # Create the masterfile name with date and time
        textfile=f"interview_{timestamp}.txt"
        # Write the chat history to the text file
        with open(textfile, "w") as f:
            for entry in st.session_state.chat_history:
                f.write(f"{entry['role']}: {entry['content']}\n")

if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    # Streamlit app title
    st.title("Candidate Interview")
    myCompare_CV_App()