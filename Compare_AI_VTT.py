import streamlit as st
from transformers import (
    pipeline, AutoProcessor,
    AutoModelForImageTextToText, AutoTokenizer, LlavaForConditionalGeneration,
    LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration,
)
from PIL import Image
import requests
import os
import time
import torch
import av
import numpy as np
from huggingface_hub import hf_hub_download
import sentencepiece
import torchvision.transforms as transforms
import cv2

def myCompare_VTT_App():
    # Streamlit app title
    st.title("Media Description with LLM")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"cuda available" if torch.cuda.is_available() else "cuda not available")
    device = torch.device("cpu")
    st.write(f"device: {device}")

    # File uploader for video or image
    uploaded_file = st.file_uploader("Choose a media file (image or video)", type=["jpg", "jpeg", "png", "bmp", "tiff", "mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:

        # Save the uploaded file temporarily
        file_path = os.path.abspath(f"temp_{uploaded_file.name}")
        st.write(f"file_path: {file_path}")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(file_path)[1].lower()
        st.write(f"ext: {ext}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

        # Create three columns
        col1, col2, col3 = st.columns([1, 1, 1])

        # Calculate responses and times for each model
        with col1:
            st.subheader("tbd")

        with col2:
            # Define the model ID and the local path where the model should be stored
            model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
            st.subheader(model_id)

            start_time = time.time()
            # Load the model and processor
            @st.cache_resource
            def load_model2():

                model = LlavaForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                ).to(device)

                processor = AutoProcessor.from_pretrained(model_id)
                return model, processor

            model, processor = load_model2()

            # Define the conversation
            # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
            # Each value in "content" has to be a list of dicts with types ("text", "image") 
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene in detail"},
                    {"type": "image"},
                    ],
                },
            ]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Determine file type and add to conversation
            if ext in image_extensions:
                raw_image = Image.open(file_path)
            

                if raw_image is None:
                    st.write("Image is not loaded properly.")
                else:
                    try:
                        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
                        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
                        end_time = time.time()
                        st.write(f"Time taken: {duration:.2f} seconds")
                        st.write(processor.decode(output[0][2:], skip_special_tokens=True))
                    except Exception as e:
                        st.write(f"Error during model inference: {e}")
                
            elif ext in video_extensions:
                
                # Function to perform inference on a video
                def infer_video(model, video_path, frame_skip=10):
                    # Öffnet das Video, das sich unter dem Pfad video_path befindet, mit OpenCV.
                    # cap ist ein Objekt, das Zugriff auf die Frames des Videos ermöglicht.
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    results = []
                    frames = []  # Liste zum Sammeln von Frames

                    for i in range(frame_count):
                        # cap.read() liest das nächste Frame aus dem Video.
                        # ret: Gibt True zurück, wenn das Frame erfolgreich gelesen wurde, sonst False.
                        # frame: Das gelesene Frame als NumPy-Array.
                        # Wenn kein Frame mehr gelesen werden kann (ret == False), wird die Schleife abgebrochen.
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Konvertiert das Frame von OpenCVs BGR-Farbformat in das RGB-Format.
                        # Wandelt das NumPy-Array in ein PIL-Bild um, das für die Verarbeitung durch das Modell geeignet ist.
                        if i % frame_skip == 0:
                            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            frames.append(frame)
                            print(f"Selected {len(frames)} frames for processing.")
                            inputs = processor(images=frame, text=prompt, return_tensors='pt').to(device, torch.float16)
                            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                            results.append(processor.decode(output[0][2:], skip_special_tokens=True))

                    cap.release()  # Gibt das Video-Objekt frei, um Ressourcen freizugeben.
                    # Gibt die Liste der Ergebnisse zurück, die während der Verarbeitung jedes Frames gesammelt wurden.
                    return frames, results

                video_frames, video_outputs  = infer_video(model, file_path, frame_skip=80)
                
                end_time = time.time()
                duration = end_time - start_time
                st.write(f"Time taken: {duration:.2f} seconds")

                for i, output in enumerate(video_outputs):
                    # Split at "assistant" and take the part after
                    if "assistant" in output:
                        response = output.split("assistant", 1)[1].strip()
                    else:
                        response = output.strip()
                    
                    # Optional: Remove the prompt "What are these?" if it was included inside the same string
                    if response.lower().startswith("what are these?"):
                        response = response[len("What are these?"):].strip()

                    st.write(f"Output {i + 1}:\n{response}\n{'-' * 80}")

            else:
                st.error(f"Unsupported file type: {file_path}")
                st.stop()

            del model
            del processor

        with col3:
            # Define the model ID and the local path where the model should be stored
            model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
            st.subheader(model_id)

            start_time = time.time()
            # Load the model and processor
            @st.cache_resource
            def load_model3():
                model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(device)
                processor = LlavaNextVideoProcessor.from_pretrained(model_id, use_fast=False)
                return model, processor

            model, processor = load_model3()

            # Define the conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the scene in detail"}
                    ],
                }
            ]
            # Determine file type and add to conversation

            if ext in image_extensions:
                try:
                    image = Image.open(file_path).convert("RGB")
                    conversation[0]["content"].append({"type": "image", "image": image})
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                    st.stop()
            elif ext in video_extensions:
                conversation[0]["content"].append({"type": "video", "path": file_path})
            else:
                st.error(f"Unsupported file type: {file_path}")
                st.stop()

            # Process the conversation
            inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

            # Generate output
            output = model.generate(**inputs, max_new_tokens=300)

            # Decode and display the output
            decoded_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            end_time = time.time()
            duration = end_time - start_time
            st.write(f"Time taken: {duration:.2f} seconds")
            for line in decoded_output:
                st.write(f"{line}\n")

            # Clean up the temporary file
            os.remove(file_path)
            del model
            del processor

if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    myCompare_VTT_App()