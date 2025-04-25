#####################################################
# Programm for llm abd further model comparison     #
# run it with 'streamlit run Compare_AI_STT.py'     #
#####################################################
# 
import streamlit as st
import whisper
import time
import os
import tempfile  # Import tempfile to handle temporary files


def myCompare_STT_App():
    
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if audio_file:

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_file.read())
            temp_audio_path = temp_audio_file.name

        try:
            # Create three columns for Speech-To-Text model comparison
            col1, col2, col3 = st.columns([1, 1, 1])

            # Process and display results for each model
            with col1:
                st.subheader("Whisper tiny      ")
                start_time = time.time()
                model = whisper.load_model("small", device="cpu")
                transcription = model.transcribe(temp_audio_path)
                del model  # Modell löschen
                end_time = time.time()
                duration = end_time - start_time
                st.write(f"Time taken: {duration:.2f} seconds")
                st.write(transcription["text"])  # Extract and display only the text field

            with col2:
                st.subheader("Whisper small      ")
                start_time = time.time()
                model = whisper.load_model("small", device="cpu")
                transcription = model.transcribe(temp_audio_path)
                del model  # Modell löschen
                end_time = time.time()
                duration = end_time - start_time
                st.write(f"Time taken: {duration:.2f} seconds")
                st.write(transcription["text"])  # Extract and display only the text field

            with col3:
                st.subheader("Whisper medium ")
                start_time = time.time()
                model = whisper.load_model("medium", device="cpu")
                transcription = model.transcribe(temp_audio_path)
                del model  # Modell löschen
                end_time = time.time()
                duration = end_time - start_time
                st.write(f"Time taken: {duration:.2f} seconds")
                st.write(transcription["text"])  # Extract and display only the text field
        finally:
            # Ensure the temporary file is deleted after processing
            os.remove(temp_audio_path)

            
if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    st.title("STT Comparison App")
    myCompare_STT_App()