# tmp_Compare_AI_CHT.py

import LIB_LLMChatLib as myllm

import os
import threading
import argparse
from typing import List, Dict, Optional # Added for type hints
import time # Added for potential delays

from dotenv import load_dotenv

# --- (Keep existing configuration loading) ---
# Pfad zur .env-Datei anpassen, falls nötig
dotenv_path = '/home/gabriel/myProject/myvenv/.env' # Beispielpfad - bitte anpassen!
if os.path.exists(dotenv_path):
    print(f"Lade Umgebungsvariablen aus {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"Warnung: .env-Datei nicht gefunden unter {dotenv_path}. Lese aus Systemumgebung.")

# Lese Konfiguration aus Umgebungsvariablen
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

# Größe des Chat-Verlaufsfensters
HISTORY_WINDOW_SIZE = 10

# --- Haupt-Chat-Logik ---

def main():
    parser = argparse.ArgumentParser(description="Paralleler Chat mit LLMs zur Videoanalyse via GCS Upload.")
    # --- (Keep existing argument parsing) ---
    parser.add_argument(
        "-f", "--file",
        help="Pfad zur lokalen Videodatei, die analysiert werden soll.",
        required=True
    )
    parser.add_argument(
        "--bucket",
        default=GCS_BUCKET_NAME, # Standardwert aus Umgebungsvariable
        help="Name des GCS Buckets für den Upload."
    )
    parser.add_argument(
        "--project",
        default=GOOGLE_CLOUD_PROJECT_ID, # Standardwert aus Umgebungsvariable
        help="Google Cloud Projekt ID für GCS."
    )
    parser.add_argument(
        "--history_size", type=int, default=HISTORY_WINDOW_SIZE,
        help=f"Anzahl letzter Nachrichten im Kontext (Standard: {HISTORY_WINDOW_SIZE})."
    )
    parser.add_argument(
        "--gemini_model", type=str, default=myllm.DEFAULT_GEMINI_MODEL,
        help=f"Gemini Modell API Identifier (Standard: {myllm.DEFAULT_GEMINI_MODEL})."
    )
    parser.add_argument(
        "--chatgpt_model", type=str, default=myllm.DEFAULT_CHATGPT_MODEL,
        help=f"ChatGPT Modell API Identifier (Standard: {myllm.DEFAULT_CHATGPT_MODEL})."
    )
    parser.add_argument( # New argument for frame extraction control
        "--max_frames_gpt", type=int, default=10,
        help="Maximale Anzahl extrahierter Frames für ChatGPT (Standard: 10)."
    )
    args = parser.parse_args()


    # --- (Keep existing input validation) ---
    local_video_file_path = args.file
    if not os.path.isfile(local_video_file_path):
        print(f"Fehler: Lokale Videodatei nicht gefunden: {local_video_file_path}")
        return

    if not args.bucket:
         print("Fehler: GCS Bucket Name nicht angegeben (weder via --bucket noch via GCS_BUCKET_NAME env var).")
         return
    if not args.project:
         print("Fehler: Google Cloud Project ID nicht angegeben (weder via --project noch via GOOGLE_CLOUD_PROJECT env var).")
         return

    # Überprüfe, ob essentielle Variablen gesetzt sind
    config_errors = []
    if not GEMINI_API_KEY: config_errors.append("GEMINI_API_KEY")
    if not OPENAI_API_KEY: config_errors.append("OPENAI_API_KEY")
    # GCS Bucket/Project checked via args now
    # if not GCS_BUCKET_NAME: config_errors.append("GCS_BUCKET_NAME")
    # if not GOOGLE_CLOUD_PROJECT_ID: config_errors.append("GOOGLE_CLOUD_PROJECT")

    if config_errors:
        print("\nFehler: Folgende Umgebungsvariablen fehlen oder sind nicht gesetzt:")
        for error in config_errors:
            print(f" - {error}")
        print(f"Bitte stelle sicher, dass sie in '{dotenv_path}' oder der Systemumgebung definiert sind.")
        exit(1) # Beende das Skript, da essentielle Konfiguration fehlt

    # --- GCS Upload (Original Video) ---
    print(f"\n--- Starte GCS Upload für Originalvideo {os.path.basename(local_video_file_path)} ---")
    gcs_video_url = myllm.upload_to_gcs_and_get_public_url(
        local_file_path=local_video_file_path,
        bucket_name=args.bucket,
        project_id=args.project,
        # destination_blob_name could be specified if needed
    )

    if not gcs_video_url:
        print("Fehler: GCS Upload des Originalvideos fehlgeschlagen. Fortfahren nicht möglich.")
        return # Exit if original upload fails, as Gemini needs it too (or path)
    else:
        print(f"--- GCS Upload Originalvideo erfolgreich: {gcs_video_url} ---")


    # --- Frame Extraction and Upload for ChatGPT ---
    print(f"\n--- Starte Frame-Extraktion für ChatGPT aus {os.path.basename(local_video_file_path)} ---")
    # Extract frames from the *local* video file
    try:
        # Limit the number of frames extracted and uploaded
        extracted_frame_paths = myllm.extract_relevant_frames(local_video_file_path) # Use default threshold for now
        if len(extracted_frame_paths) > args.max_frames_gpt:
             print(f"Info: Reduziere Anzahl Frames von {len(extracted_frame_paths)} auf {args.max_frames_gpt}")
             # Simple sampling: take evenly spaced frames
             indices = np.linspace(0, len(extracted_frame_paths) - 1, args.max_frames_gpt, dtype=int)
             relevant_frame_paths = [extracted_frame_paths[i] for i in indices]
        else:
             relevant_frame_paths = extracted_frame_paths

        print(f"Info: {len(relevant_frame_paths)} relevante Frames extrahiert.")

    except Exception as e:
        print(f"Fehler bei der Frame-Extraktion: {e}. ChatGPT-Analyse wird ohne Bilder versucht.")
        relevant_frame_paths = [] # Ensure it's an empty list

    chatgpt_frame_urls = []
    if relevant_frame_paths:
        print(f"\n--- Starte GCS Upload für {len(relevant_frame_paths)} Frames ---")
        # Create a unique subdirectory for frames in GCS
        video_base_name = os.path.splitext(os.path.basename(local_video_file_path))[0]
        gcs_frame_dir = f"extracted_frames/{video_base_name}"

        for i, frame_path in enumerate(relevant_frame_paths):
            frame_basename = os.path.basename(frame_path)
            destination_blob = f"{gcs_frame_dir}/{frame_basename}"
            print(f"Info [GCS Frames]: Uploading Frame {i+1}/{len(relevant_frame_paths)}: {frame_basename}...")
            frame_url = myllm.upload_to_gcs_and_get_public_url(
                local_file_path=frame_path,
                bucket_name=args.bucket,
                project_id=args.project,
                destination_blob_name=destination_blob
            )
            if frame_url:
                chatgpt_frame_urls.append(frame_url)
            else:
                print(f"Warnung [GCS Frames]: Upload für Frame {frame_path} fehlgeschlagen.")
            # Optional: Add a small delay to avoid hitting rate limits if many frames
            # time.sleep(0.1)

        if chatgpt_frame_urls:
            print(f"--- GCS Upload für {len(chatgpt_frame_urls)} Frames abgeschlossen. ---")
        else:
            print("Warnung: Keine Frames erfolgreich zu GCS hochgeladen. ChatGPT erhält nur Text.")
        # Clean up local frames? Optional.
        # for frame_path in relevant_frame_paths:
        #     try: os.remove(frame_path)
        #     except OSError: pass
        # try: os.rmdir(os.path.dirname(relevant_frame_paths[0])) # Remove the dir if empty
        # except (OSError, IndexError): pass


    # --- LLM Initialisierung ---
    print("\nStarte LLM-Initialisierung...")
    try:
        llm1 = myllm.GeminiLLM(api_key=GEMINI_API_KEY, model_name=args.gemini_model)
        llm2 = myllm.ChatGPTLLM(api_key=OPENAI_API_KEY, model_name=args.chatgpt_model)
        print("LLM-Clients erfolgreich initialisiert.")
    except (ValueError, ImportError, Exception) as e:
        print(f"\nFehler bei LLM-Initialisierung: {e}")
        print("Stelle sicher, dass API Keys korrekt sind und Bibliotheken (google-generativeai, openai, google-cloud-storage, opencv-python, numpy) installiert/aktuell sind.")
        return

    llm_map = {"Gemini": llm1, "ChatGPT": llm2}

    print("\nStarte parallelen Chat...")
    # ... (rest of the setup: model names, history size, initial context) ...
    print(f"Verwende Gemini Model: {args.gemini_model}")
    print(f"Verwende ChatGPT Model: {args.chatgpt_model}")
    print(f"Chat History Größe: {args.history_size}")

    # --- Chat-Verlauf und Kontext ---
    chat_history: List[Dict[str, str]] = []
    initial_text_context = None # Kein separater Text-Kontext in diesem Beispiel

    print("\nChat gestartet. Erste Anfrage wird gesendet. Gib 'quit' ein, um zu beenden.")
    first_request = True

    while True:
        if first_request:
            # Updated initial prompt to mention frames if available
            prompt_detail = f"(Original Video URL: {gcs_video_url or 'URL nicht verfügbar'})"
            if chatgpt_frame_urls:
                 prompt_detail += f"\nAnalyse basiert auf {len(chatgpt_frame_urls)} extrahierten Frames."

            user_input = f"""
AI Video Analyse Prompt: Kritische Sicherheitsereignisse
Ziel: Analysiere das bereitgestellte Sicherheitskamera-Videomaterial {prompt_detail} ausschließlich auf Anzeichen von Einbruch/Gewaltsames Eindringen und Feuer. Generiere prägnante, sofort verwertbare Alarme NUR für diese Ereignisse.
Eingabe: Das Video (direkt hochgeladen für Gemini) oder extrahierte Frames (via URLs für ChatGPT).
Zu detektierende kritische Trigger:

Einbruch / Gewaltsames Eindringen:
    Sichtbare Beschädigung von Türen, Fenstern, Schlössern (z.B. Glasbruch, aufgehebelter Rahmen).
    Personen, die versuchen, auf unübliche Weise einzudringen (z.B. Klettern, Einschlagen).
    Verdächtiges Hantieren direkt an potenziellen Eintrittspunkten (Tür/Fenster).
Feuer:
    Sichtbarer Rauch (möglichst von Nebel/Dampf unterscheiden).
    Sichtbare Flammen.

Ausgabeformat (NUR bei erkanntem Trigger):
KRITISCHER ALARM
Timestamp: [Ca. Zeit im Video oder Frame-Bezug, falls bestimmbar]
Ereignis: [Einbruchsversuch / Feuer]
Ort: [Kurze Beschreibung, z.B. 'Vordertür', 'Fenster links', 'Dachbereich']
Evidenz: [Sehr kurze Beschreibung des Triggers, z.B. 'Scheibe eingeschlagen', 'Rauch sichtbar', 'Tür wird aufgehebelt']

Anweisungen:
Ignoriere jegliche Normalaktivität: Personen gehen vorbei, autorisierte Eintritte, Lieferungen, Wettereffekte (außer bei extremer Sichtbehinderung), Tiere etc.
Melde KEINE allgemeinen 'verdächtigen Aktivitäten', die nicht den oben genannten Triggern entsprechen.
Priorisiere Eindeutigkeit und Schnelligkeit.
Wenn keine kritischen Ereignisse gemäß den Triggern erkannt werden, gib NUR aus: STATUS: Keine kritischen Ereignisse detektiert.
Füge eine kurze Beschreibung des Videos/der Frames beim Fokussieren auf die Bewegungen hinzu.
"""
            print("\nSende initiale Videoanalyse-Anfrage...")
        else:
            # Eingabe für Folgefragen erhalten
            try:
                user_input = input("\nDu (weitere Fragen zum Video/Frames oder neuer Prompt): ")
            except EOFError:
                print("\nEingabe beendet.")
                break

        if user_input.lower() == 'quit':
            print("Chat wird beendet.")
            break

        if not user_input.strip():
            continue

        # Benutzereingabe zum Verlauf hinzufügen
        chat_history.append({"role": "user", "content": user_input})

        # Relevantes Verlaufsfenster holen (nur Text)
        windowed_history = myllm.get_windowed_history(chat_history[:-1], args.history_size)

        # --- Threads erstellen und starten ---
        results = {"Gemini": None, "ChatGPT": None}
        threads: List[threading.Thread] = []
        print("Sende Anfrage an LLMs (kann bei Video länger dauern)...")

        # --- Prepare inputs for threads ---
        # Gemini: Still uses local video path for direct upload
        gemini_video_input = {"video_path": local_video_file_path}

        # ChatGPT: Uses the list of frame URLs now
        # Pass None if no frames were successfully uploaded
        chatgpt_video_input = {"frame_urls": chatgpt_frame_urls if chatgpt_frame_urls else None}

        # Verwende die angepasste Thread-Runner-Funktion
        thread_gemini = threading.Thread(
            target=run_llm_in_thread_extended,
            args=(llm_map["Gemini"], user_input, windowed_history, initial_text_context,
                  gemini_video_input, # Pass Gemini's input dict
                  results, "Gemini"),
            daemon=True
        )
        threads.append(thread_gemini)

        thread_chatgpt = threading.Thread(
            target=run_llm_in_thread_extended,
            args=(llm_map["ChatGPT"], user_input, windowed_history, initial_text_context,
                  chatgpt_video_input, # Pass ChatGPT's input dict (frame URLs)
                  results, "ChatGPT"),
            daemon=True
        )
        threads.append(thread_chatgpt)

        thread_gemini.start()
        thread_chatgpt.start()

        # Auf Threads warten
        for thread in threads:
            thread.join()

        print("-" * 20)

        # --- Ergebnisse ausgeben ---
        gemini_response = results.get("Gemini", "[Gemini: Kein Ergebnis - Thread-Problem?]")
        chatgpt_response = results.get("ChatGPT", "[ChatGPT: Kein Ergebnis - Thread-Problem?]")

        print(f"Gemini ({args.gemini_model}):\n{gemini_response}")
        print("-" * 10)
        print(f"ChatGPT ({args.chatgpt_model}):\n{chatgpt_response}")
        print("-" * 20)

        # --- Verlauf aktualisieren (nur Text-Antworten) ---
        # Ensure responses are strings before adding
        gemini_resp_str = str(gemini_response) if gemini_response is not None else ""
        chatgpt_resp_str = str(chatgpt_response) if chatgpt_response is not None else ""
        chat_history.append({"role": "model", "content": gemini_resp_str}) # Gemini -> model
        chat_history.append({"role": "assistant", "content": chatgpt_resp_str}) # ChatGPT -> assistant

        # Nach der ersten Anfrage ist es keine erste Anfrage mehr
        first_request = False
        # After the first request, don't send frames again unless explicitly asked
        # or if you want to keep context. For simplicity, we clear them.
        chatgpt_frame_urls = [] # Don't resend frames on subsequent turns

# --- Angepasste Thread Runner Funktion ---
# Modify to accept different video input types
def run_llm_in_thread_extended(llm_instance: myllm.LLMInterface,
                               prompt: str,
                               history: List[Dict[str, str]],
                               initial_text_context: Optional[str],
                               video_input: Dict, # Accepts Dict for video/frame info
                               results: Dict,
                               key: str):
    """Führt die LLM-Chat-Funktion (mit flexibler Video/Frame-Input) in einem Thread aus."""
    try:
        # Pass the whole dictionary; the chat method will extract what it needs
        response = llm_instance.chat(prompt, history, initial_text_context,
                                     video_data=video_input) # Pass the dict

        results[key] = response
    except Exception as e:
        print(f"Fehler im Thread für {key}: {type(e).__name__} - {e}")
        # Füge Traceback hinzu für Debugging
        import traceback
        print(traceback.format_exc())
        results[key] = f"[{llm_instance.__class__.__name__} Thread Error: {e}]"


# --- Skriptstart ---
if __name__ == "__main__":
    # Add numpy import needed for frame limiting
    import numpy as np
    main()