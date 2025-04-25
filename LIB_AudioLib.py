import pyaudio
import numpy as np
import os
import subprocess
import wave
import time
import audioop

from scipy.signal import resample

from langchain.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Specify the path to your .env file
# Load environment variables from .env file
dotenv_path = '/home/gabriel/myProject/myvenv/.env' # <<< PRÜFE DIESEN PFAD GENAU!
if os.path.exists(dotenv_path):
    print(f"Lade Umgebungsvariablen aus {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print(f"Warnung: .env-Datei nicht gefunden unter {dotenv_path}.")

# --------------------------------------------------------------------------------------------
def get_audio_output_devices():
    """
    This function lists all available audio output devices on the system.
    
    It uses the pyaudio library to query the available audio devices and 
    filters out the output devices. The function returns a list of dictionaries 
    containing the device name and its corresponding index.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'name' and 'index' keys.
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Initialize an empty list to store output devices
    output_devices = []

    # Get the number of available devices
    device_count = p.get_device_count()

    # Get the default output device
    try:
        default_device_index = p.get_default_output_device_info()['index']
    except IOError:
        default_device_index = None
    
    # Iterate over the devices and filter out the output devices
    for idx in range(device_count):
        device_info = p.get_device_info_by_index(idx)
        if device_info['maxOutputChannels'] > 0:
            output_devices.append({
                'name': device_info['name'],
                'index': idx
            })
            if "plantronic" in device_info['name'].lower():
                default_device_index = idx
    # Terminate the PyAudio instance
    p.terminate()
    
    return output_devices, default_device_index

# --------------------------------------------------------------------------------------------
def is_sample_rate_supported(device_index, sample_rate):
    """
    Check if the given sample rate is supported by the specified device.
    
    Args:
        device_index (int): The index of the audio output device.
        sample_rate (int): The sample rate to check.
    
    Returns:
        bool: True if the sample rate is supported, False otherwise.
    """
    p = pyaudio.PyAudio()
    try:
        if device_index is None:
            device_index = p.get_default_input_device_info()['index']
        device_info = p.get_device_info_by_index(device_index)
        supported = p.is_format_supported(sample_rate,
                                          output_device=device_index,
                                          output_channels=1,
                                          output_format=pyaudio.paInt16)
    except ValueError:
        supported = False
    p.terminate()
    return supported

# --------------------------------------------------------------------------------------------
def generate_sound_on_device(device_index, duration=1, frequency=440, sample_rate=48000):
    """
    This function generates a sound on a specified audio output device for a given duration.
    
    Args:
        device_index (int): The index of the audio output device.
        duration (float): The duration of the sound in seconds. Default is 1 second.
        frequency (float): The frequency of the sound in Hz. Default is 440 Hz (A4 note).
        sample_rate (int): The sample rate in Hz. Default is 448000 Hz.
    
    Returns:
        None
    """
    if not is_sample_rate_supported(device_index, sample_rate):
        raise ValueError(f"Sample rate {sample_rate} is not supported by device {device_index}")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Generate the sound wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert the wave to 16-bit PCM format
    wave = (wave * 32767).astype(np.int16)
    
    # Open the audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    output_device_index=device_index)
    
    # Play the sound
    stream.write(wave.tobytes())
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate the PyAudio instance
    p.terminate()

# --------------------------------------------------------------------------------------------
def capture_speech_from_microphone(duration=5, sample_rate=48000, channels=1, output_filename="output.wav", silence_threshold=500, silence_duration=10):
    """
    This function captures audio from the microphone and saves it to a WAV file.
    If duration is set to 0, it records until silence is detected for more than `silence_duration` seconds.
    It returns the real duration of the speech (excluding trailing silence).
    
    Args:
        duration (float): The duration of the recording in seconds. If 0, records until silence is detected.
        sample_rate (int): The sample rate in Hz. Default is 48000 Hz.
        channels (int): The number of audio channels. Default is 1 (mono).
        output_filename (str): The name of the output WAV file. Default is "output.wav".
        silence_threshold (int): The RMS threshold to detect silence. Default is 500.
        silence_duration (int): The duration of silence (in seconds) to stop recording when duration=0. Default is 10 seconds.
    
    Returns:
        float: The real duration of the speech in seconds (excluding trailing silence).
    """
    if not is_sample_rate_supported(None, sample_rate):
        raise ValueError(f"Sample rate {sample_rate} is not supported by the default input device")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open the audio stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    
    print("Recording...")
    
    # Initialize an empty list to store the recorded frames
    frames = []
    silence_start_time = None  # Track when silence starts
    speech_start_time = time.time()  # Track when recording starts
    last_speech_time = speech_start_time  # Track the last time speech was detected
    
    if duration == 0:
        # Record endlessly until silence is detected
        while True:
            data = stream.read(1024)
            frames.append(data)
            
            # Calculate the RMS (root mean square) of the audio data
            rms = audioop.rms(data, 2)  # 2 bytes per sample (16-bit audio)
            
            if rms < silence_threshold:
                # Silence detected
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > silence_duration:
                    # Stop recording if silence lasts longer than `silence_duration`
                    print("Silence detected.")
                    break
            else:
                # Reset silence timer if sound is detected
                silence_start_time = None
                last_speech_time = time.time()  # Update the last speech time
    else:
        # Record for the specified duration
        num_frames = int(sample_rate / 1024 * duration)
        for _ in range(num_frames):
            data = stream.read(1024)
            frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate the PyAudio instance
    p.terminate()
    
    # Save the recorded frames to a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    # Calculate the real duration of the speech
    real_duration = last_speech_time - speech_start_time
    return real_duration

# --------------------------------------------------------------------------------------------
def speech_to_text(duration=5, sample_rate=48000):
    """
    This function captures audio from the microphone for a specified duration and converts it to text.
    
    Args:
        duration (float): The duration of the recording in seconds. Default is 5 seconds.
        sample_rate (int): The sample rate in Hz. Default is 48000 Hz.
    
    Returns:
        str: The transcribed text from the audio.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=sample_rate) as source:
        print("Recording...")
        audio = recognizer.record(source, duration=duration)
        print("Recording finished.")
    
    try:
        # Use Sphinx recognizer to recognize the audio
        text = recognizer.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        return "Sphinx could not understand the audio"
    except sr.RequestError as e:
        return f"Sphinx error; {e}"

# --------------------------------------------------------------------------------------------
def play_speech_on_device(file_path, device_index):
    """
    This function plays a sound file on a specified audio output device.
    
    Args:
        file_path (str): The path to the sound file (WAV format).
        device_index (int): The index of the audio output device.
    
    Returns:
        None
    """
    # Open the sound file
    wf = wave.open(file_path, 'rb')
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open the audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=device_index)
    
    # Read data from the sound file
    data = wf.readframes(1024)
    
    # Play the sound
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate the PyAudio instance
    p.terminate()

# --------------------------------------------------------------------------------------------
def pico_text_to_speech(text, output_filename="tts_output.wav", language='de-DE'):
    """
    This function converts text to speech using Pico TTS and saves it to a WAV file.
    
    Args:
        text (str): The text to convert to speech.
        output_filename (str): The name of the output WAV file. Default is "tts_output.wav".
    
    Returns:
        None
    """
    # Use Pico TTS to convert text to speech and save it to a WAV file
    command = ['pico2wave', '--wave', output_filename, "-l", language, text]
    subprocess.run(command, check=True)

# --------------------------------------------------------------------------------------------

async def upload_file(file_path):
    """
    Upload a file as a data source for the LLM.
    Args:
        file_path (str): The path to the file to upload.
    Returns:
        LocalFileLoader: The loaded file.
    """
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return loader.load()

# --------------------------------------------------------------------------------------------
def ask_question(documents, question):
    """
    Ask a question to the LLM using the uploaded documents.
    Args:
        documents (List[Document]): The list of documents to use as context.
        question (str): The question to ask.
    Returns:
        str: The response from the LLM.
    """
   
    # Access the API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm=llm, chain_type='stuff')
    response = chain.run(input_documents=documents, question=question)
    return response
# --------------------------------------------------------------------------------------------
def print_response(response):
    """
    Print the response from the LLM.
    Args:
        response (str): The response to print.
    Returns:
        None
    """
    print("Response:", response)

# --------------------------------------------------------------------------------------------

def convert_audio_params(frames, original_params, target_params):
    # Konvertieren der Audiodaten, um sie an die Zielparameter anzupassen
    audio_array = np.frombuffer(frames, dtype=np.int16)

    # Anpassen der Abtastrate
    if original_params.framerate != target_params.framerate:
        # Hier könnte eine Bibliothek wie `librosa` oder `scipy` verwendet werden, um die Abtastrate zu ändern
        # Resample the audio to the target framerate
        num_samples = int(len(audio_array) * target_params.framerate / original_params.framerate)
        audio_array = resample(audio_array, num_samples).astype(np.int16)

    # Anpassen der Anzahl der Kanäle
    if original_params.nchannels != target_params.nchannels:
        if target_params.nchannels == 1:
            # Mono: Mittelwert der Kanäle
            audio_array = np.mean(audio_array.reshape(-1, original_params.nchannels), axis=1).astype(np.int16)
        elif target_params.nchannels == 2:
            # Stereo: Duplizieren des Monokanals oder Beibehalten der Stereokanäle
            if original_params.nchannels == 1:
                audio_array = np.repeat(audio_array, 2)

    # Anpassen der Sample-Breite (z.B. 16-bit auf 8-bit)
    if original_params.sampwidth != target_params.sampwidth:
        if target_params.sampwidth == 1:
            audio_array = (audio_array >> 8).astype(np.int8)
        elif target_params.sampwidth == 2:
            audio_array = (audio_array << 8).astype(np.int16)

    return audio_array.tobytes()

# --------------------------------------------------------------------------------------------
def my_wav_aggregator(masterfile, addfile):
    # Öffnen der Master-Datei, falls vorhanden
    if masterfile and os.path.exists(masterfile):
        with wave.open(masterfile, 'rb') as master_wav:
            master_frames = master_wav.readframes(master_wav.getnframes())
            master_params = master_wav.getparams()
    else:
        master_frames = b''
        master_params = None
        masterfile = addfile

    # Öffnen der hinzuzufügenden Datei
    with wave.open(addfile, 'rb') as add_wav:
        add_frames = add_wav.readframes(add_wav.getnframes())
        add_params = add_wav.getparams()

        # Wenn die Master-Datei nicht existiert, verwenden wir die Parameter der hinzuzufügenden Datei
        if master_params is None:
            master_params = add_params
            master_frames = add_frames
        else:
            # Anpassen der Parameter der hinzuzufügenden Datei an die Master-Datei
            if add_params.nchannels != master_params.nchannels or \
               add_params.sampwidth != master_params.sampwidth or \
               add_params.framerate != master_params.framerate:
                add_frames = convert_audio_params(add_frames, add_params, master_params)

    # Zusammenführen der Audiodaten
    master_frames += add_frames

    # Schreiben der kombinierten Daten in die Master-Datei
    with wave.open(masterfile, 'wb') as combined_wav:
        combined_wav.setparams(master_params)
        combined_wav.writeframes(master_frames)

    return masterfile

# --------------------------------------------------------------------------------------------