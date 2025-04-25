# LIB_LLMChatLib.py

# --- (Keep existing imports: os, mimetypes, base64, pathlib, time, pickle, cv2, numpy, abc, typing, quote) ---
import os
import mimetypes # Für Dateitypen
import base64   # Früher für OpenAI verwendet, jetzt weniger relevant für Video
import pathlib  # Für Pfad-Operationen
import time     # Für Wartezeiten (Gemini Upload)
import pickle   # Für Drive-Token (falls Drive-Funktionen hinzugefügt werden)
import cv2      # <<< Make sure opencv-python is installed: pip install opencv-python
import numpy as np # <<< Make sure numpy is installed: pip install numpy
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union # Union added
from urllib.parse import quote # Für GCS URL Encoding


# --- (Keep existing Google Cloud/OpenAI imports and checks) ---
try:
    import google.generativeai as genai
except ImportError:
    print("WARNUNG: google-generativeai nicht installiert. Gemini LLM wird nicht funktionieren. -> pip install google-generativeai")
    genai = None

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, Forbidden
except ImportError:
    print("WARNUNG: google-cloud-storage nicht installiert. GCS Upload/Frame Upload wird nicht funktionieren. -> pip install google-cloud-storage")
    storage = None

try:
    from openai import OpenAI, APIError, APITimeoutError, BadRequestError
except ImportError:
     print("WARNUNG: openai nicht installiert. ChatGPT LLM wird nicht funktionieren. -> pip install --upgrade openai")
     OpenAI = None
     # Define dummy error classes if openai is not installed
     class APIError(Exception): pass
     class APITimeoutError(Exception): pass
     class BadRequestError(Exception): pass

# --- (Keep existing DEFAULT_GEMINI_MODEL, DEFAULT_CHATGPT_MODEL) ---
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"
DEFAULT_CHATGPT_MODEL = "gpt-4o"

# --- LLM Abstraktion (Framework) ---

class LLMInterface(ABC):
    """Abstrakte Basisklasse für LLM-Interaktionen."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError(f"API key for {self.__class__.__name__} is missing.")
        self.api_key = api_key
        self.model_name = model_name
        print(f"Info: {self.__class__.__name__} initialisiert für Modell '{self.model_name}'.")

    @abstractmethod
    def chat(self,
             prompt: str,
             history: List[Dict[str, str]],
             initial_text_context: Optional[str] = None,
             # Use a dictionary to pass video/frame data flexibly
             video_data: Optional[Dict[str, Union[str, List[str]]]] = None
             ) -> str:
        """
        Sendet eine Nachricht (und optional Video/Frames) an das LLM.
        Implementierungen extrahieren die benötigten Daten aus video_data.
        """
        pass

    def _prepare_history(self, history: List[Dict[str, str]], api_role_user: str, api_role_model: str) -> List[Dict[str, str]]:
        """Bereitet Text-History für die jeweilige API vor."""
        # (Keep existing implementation)
        formatted_history = []
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            # Ignoriere Nachrichten ohne Inhalt oder mit komplexem Inhalt (für diese einfache History)
            if isinstance(content, str) and content.strip():
                if role == "user":
                    formatted_history.append({"role": api_role_user, "content": content})
                elif role in ["model", "assistant"]:
                    formatted_history.append({"role": api_role_model, "content": content})
        return formatted_history


# --- Konkrete LLM Implementierungen ---

class GeminiLLM(LLMInterface):
    """Implementierung für Google Gemini mit Video-Upload."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_MODEL):
        # (Keep existing implementation)
        if not genai:
             raise ImportError("Modul 'google.generativeai' nicht geladen. Installation prüfen.")
        super().__init__(api_key, model_name)
        try:
            genai.configure(api_key=self.api_key)
            # Modell-Instanziierung
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Info: Gemini Client für Modell '{self.model_name}' erfolgreich konfiguriert.")
        except Exception as e:
            print(f"Fehler bei Initialisierung/Konfiguration des Gemini Models: {e}")
            raise

    def chat(self,
             prompt: str,
             history: List[Dict[str, str]],
             initial_text_context: Optional[str] = None,
             video_data: Optional[Dict[str, Union[str, List[str]]]] = None
             ) -> str:

        print(f"DEBUG [Gemini]: Starte Chat. Video Path: {video_data.get('video_path') if video_data else 'None'}") # Debug 1
        # Initialisiere uploaded_file hier mit None, um sicherzustellen, dass es existiert
        uploaded_file = None
        # Extract video_path from the dictionary
        video_path = video_data.get("video_path") if video_data else None
        # video_url could also be extracted if needed, but path is preferred here
        video_url = video_data.get("video_url") if video_data else None # Added for consistency

        print(f"Info [Gemini]: Empfange Prompt. Video Path: {'Ja' if video_path else 'Nein'}, Video URL: {'Ja' if video_url else 'Nein'}")

        # (Keep existing history preparation)
        api_history = self._prepare_history(history, api_role_user="user", api_role_model="model")

        # (Keep existing prompt/context combination)
        full_prompt_text = prompt
        if initial_text_context:
            full_prompt_text = f"System Context:\n---\n{initial_text_context}\n---\n\nUser Prompt:\n{prompt}"

        # (Keep existing 'contents' list build from history)
        contents = []
        for message in api_history:
             contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})


        # --- Prepare current user message (Text + Video) ---
        current_user_parts = [{"text": full_prompt_text}]
        video_processed = False

        # Prioritize direct upload via video_path
        if video_path:
            print(f"DEBUG [Gemini]: Video Path gefunden: {video_path}") # Debug 2
            print(f"Info [Gemini]: Priorisiere direkten Upload via video_path: {video_path}")
            # --- (Keep the existing Gemini video upload and polling logic) ---
            video_file_path = pathlib.Path(video_path)
            if video_file_path.is_file():
                mime_type, _ = mimetypes.guess_type(video_file_path)
                if mime_type and mime_type.startswith("video/"):
                    print(f"DEBUG [Gemini]: Starte Upload für {mime_type}...") # Debug 3
                    print(f"Info [Gemini]: Lade Video '{video_path}' (Typ: {mime_type}) hoch...")
                    uploaded_file = None
                    try:
                        uploaded_file = genai.upload_file(path=video_file_path, display_name=video_file_path.name)
                        print(f"DEBUG [Gemini]: Upload gestartet, File Name: {uploaded_file.name}") # Debug 4
                        print(f"Info [Gemini]: Video hochgeladen. File Name: {uploaded_file.name}. Warte auf Aktivierung...")
                        polling_interval_seconds = 5
                        max_wait_seconds = 300 # Consider making this configurable
                        wait_time = 0
                        while uploaded_file.state.name != "ACTIVE" and wait_time < max_wait_seconds:
                            print(f"DEBUG [Gemini]: Status: {uploaded_file.state.name}. Warte...") # Debug 5
                            print(f"Info [Gemini]: File state ist {uploaded_file.state.name}. Warte {polling_interval_seconds}s...")
                            time.sleep(polling_interval_seconds)
                            wait_time += polling_interval_seconds
                            uploaded_file = genai.get_file(name=uploaded_file.name) # Status neu abfragen

                        if uploaded_file.state.name == "ACTIVE":
                            print(f"DEBUG [Gemini]: Video ACTIVE. URI: {uploaded_file.uri}") # Debug 6
                            print(f"Info [Gemini]: Video ist jetzt ACTIVE (URI: {uploaded_file.uri}).")
                            current_user_parts.append({"file_data": {
                                "mime_type": uploaded_file.mime_type,
                                "file_uri": uploaded_file.uri
                            }})
                            video_processed = True
                            print("DEBUG [Gemini]: file_data zu current_user_parts hinzugefügt.") # Debug 7
                        else:
                            print(f"Fehler [Gemini]: Datei '{uploaded_file.name}' erreichte nach {max_wait_seconds}s nicht den ACTIVE State (State: {uploaded_file.state.name}). Breche ab.")
                            # Clean up the file potentially?
                            # try: genai.delete_file(uploaded_file.name)
                            # except Exception: pass
                            return f"[Gemini API Error: File did not become ACTIVE after {max_wait_seconds}s (State: {uploaded_file.state.name})]"
                    except Exception as e:
                        print(f"Fehler [Gemini]: Fehler beim Hochladen oder Warten auf die Datei '{video_path}': {e}")
                        return f"[Gemini API Error: File upload/processing failed - {e}]"
                else:
                    print(f"Warnung [Gemini]: Konnte keinen Video-MIME-Typ für '{video_path}' bestimmen oder Datei ist kein Video. Ignoriere Datei.")
            else:
                 print(f"Warnung [Gemini]: Videodatei (via Pfad) '{video_path}' nicht gefunden. Ignoriere Datei.")
        elif video_url:
            # Only mention if URL was provided but path was preferred/used
            print("Info [Gemini]: Video URL ignoriert, da direkter Upload via Pfad bevorzugt wird.")

        # (Keep existing logic: add current_user_parts to contents)
        contents.append({"role": "user", "parts": current_user_parts})
        print(f"DEBUG [Gemini]: Sende folgende Contents an API: {contents}") # Debug 8 (SEHR WICHTIG!)


        if not contents:
             return "[Gemini Error: Kein Inhalt zum Senden nach Verarbeitung]"

        # --- (Keep existing API call and response handling) ---
        try:
            print(f"Info [Gemini]: Sende Anfrage an Modell '{self.model_name}'...")
            request_options = {"timeout": 600} # Timeout für Video-Verarbeitung
            # Handle potential empty response or generation config issues
            # generation_config = genai.types.GenerationConfig( # Example if needed
            #     candidate_count=1,
            #     # stop_sequences=['...'], # If needed
            #     # max_output_tokens=..., # If needed
            #     # temperature=..., # If needed
            # )
            response = self.model.generate_content(contents, request_options=request_options) # Add generation_config=generation_config if used

            print("Info [Gemini]: Antwort erfolgreich empfangen.")
            # Enhanced check for response content
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 return response.text # Accessing .text handles simple text extraction
            else:
                 # Try to get more detailed feedback
                 feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = "N/A"
                 finish_reason = "N/A"
                 safety_ratings_str = "N/A"
                 if feedback:
                     block_reason = getattr(feedback, 'block_reason', 'N/A')
                     safety_ratings = getattr(feedback, 'safety_ratings', [])
                     safety_ratings_str = ", ".join([f"{r.category}: {r.probability}" for r in safety_ratings]) if safety_ratings else "N/A"

                 # Check finish reason on the candidate if available
                 if response.candidates:
                    finish_reason = getattr(response.candidates[0], 'finish_reason', 'N/A')


                 print(f"Warnung [Gemini]: Leere, blockierte oder unvollständige Antwort. Block Reason: {block_reason}, Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings_str}")
                 error_text = f"[Gemini Error: Response Issue - Block: {block_reason}, Finish: {finish_reason}]"
                 # Optionally include the full response text for debugging, but be careful with size/PII
                 # try: error_text += f"\nFull Response: {response}"
                 # except Exception: pass
                 return error_text

        except Exception as e:
            # --- (Keep existing Gemini Exception Handling) ---
            print(f"Fehler [Gemini]: Fehler beim Aufruf von generate_content: {type(e).__name__} - {e}")
            error_details = getattr(e, 'message', str(e))
            if "DeadlineExceeded" in str(e) or "timeout" in str(e).lower():
                 return f"[Gemini API Error: Timeout überschritten. Video-Verarbeitung zu lang?]"
            if "FailedPrecondition" in str(e) and "ACTIVE state" in str(e):
                 return f"[Gemini API Error: FailedPrecondition - Datei nicht ACTIVE (Problem bei Ausführung?): {error_details}]"
            # Handle Resource Exhausted (quota)
            if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                 return f"[Gemini API Error: Quota Exceeded - {error_details}]"
            # Handle Invalid Argument (often bad input format)
            if "InvalidArgument" in str(e):
                 return f"[Gemini API Error: Invalid Argument - Prüfe Input Format/Struktur. Details: {error_details}]"

            if uploaded_file and uploaded_file.name: # <<< Korrigiert
                 try: genai.delete_file(uploaded_file.name); print(f"DEBUG [Gemini]: Datei {uploaded_file.name} nach Fehler gelöscht.") # <<< Korrigiert
                 except Exception as del_e: print(f"WARNUNG [Gemini]: Löschen nach Fehler fehlgeschlagen: {del_e}")

            import traceback
            print(traceback.format_exc())
            return f"[Gemini API Error: {type(e).__name__} - {error_details}]"


class ChatGPTLLM(LLMInterface):
    """Implementierung für OpenAI ChatGPT mit Frame-URL-Unterstützung."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_CHATGPT_MODEL):
        # (Keep existing implementation)
        if not OpenAI:
             raise ImportError("Modul 'openai' nicht geladen. Installation prüfen.")
        super().__init__(api_key, model_name)
        try:
            self.client = OpenAI(api_key=self.api_key)
            # Speichere Fehlerklassen für spezifisches Handling
            self.APIError = APIError
            self.APITimeoutError = APITimeoutError
            self.BadRequestError = BadRequestError
            print(f"Info: OpenAI Client für Modell '{self.model_name}' erfolgreich initialisiert.")
        except Exception as e:
            print(f"Fehler bei Initialisierung des OpenAI Clients: {e}")
            raise

    def chat(self,
             prompt: str,
             history: List[Dict[str, str]],
             initial_text_context: Optional[str] = None,
             # Accept the dictionary
             video_data: Optional[Dict[str, Union[str, List[str]]]] = None
             ) -> str:

        # Extract frame_urls from the dictionary
        frame_urls = video_data.get("frame_urls") if video_data else None
        # video_path = video_data.get("video_path") if video_data else None # Can ignore path here

        print(f"Info [ChatGPT]: Empfange Prompt. Frame URLs: {'Ja (' + str(len(frame_urls)) + ')' if frame_urls else 'Nein'}")

        # (Keep existing history and context preparation)
        messages = []
        if initial_text_context:
            messages.append({"role": "system", "content": initial_text_context})

        api_history = self._prepare_history(history, api_role_user="user", api_role_model="assistant")
        messages.extend(api_history)

        # --- Aktuelle User-Nachricht vorbereiten ---
        # Start with text part
        current_user_content: List[Dict[str, Union[str, Dict]]] = [{"type": "text", "text": prompt}]

        # --- Add frame URLs if available ---
        if frame_urls:
            print(f"Info [ChatGPT]: Füge {len(frame_urls)} Frame-URLs zur Anfrage hinzu...")
            for url in frame_urls:
                 # Append each frame URL using the "image_url" type
                 current_user_content.append({
                     "type": "image_url",
                     "image_url": {
                         "url": url,
                         # Consider adding detail level if needed, 'auto' or 'low' might save tokens/cost
                         # "detail": "auto"
                     }
                 })
        # else:
             # If no frames, maybe mention it if a video path was ignored?
             # if video_path: print("Warnung [ChatGPT]: Kein Frame-URLs bereitgestellt. Sende nur Text.")

        # Füge die (potenziell multimodale) User-Nachricht hinzu
        messages.append({"role": "user", "content": current_user_content})
        # --- Ende User-Nachricht ---

        # --- API Aufruf ---
        try:
            print(f"Info [ChatGPT]: Sende Anfrage an Modell '{self.model_name}'...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1500,  # Adjust as needed, maybe higher for many frames
                timeout=600.0     # Adjust as needed
            )

            print("Info [ChatGPT]: Antwort erfolgreich empfangen.")
            if response.choices:
                finish_reason = response.choices[0].finish_reason
                response_content = response.choices[0].message.content
                if finish_reason != 'stop':
                    print(f"Warnung [ChatGPT]: Finish Reason: {finish_reason} (Antwort könnte unvollständig sein)")
                # Handle potential empty content even with stop reason
                if response_content is None:
                     print("Warnung [ChatGPT]: Antwort-Inhalt ist None trotz erfolgreicher Antwort.")
                     return "[ChatGPT Error: Leerer Inhalt in der Antwort erhalten]"
                return response_content.strip()
            else:
                print("Warnung [ChatGPT]: Keine 'choices' in der Antwort.")
                # Check for other potential info in response object
                response_data = response.model_dump_json(indent=2) # Get full response as JSON string
                print(f"Debug [ChatGPT]: Vollständige Antwort (falls verfügbar):\n{response_data}")
                return "[ChatGPT Error: No response choices received]"

        except self.APITimeoutError:
             print(f"Fehler [ChatGPT]: Request timed out.")
             return f"[ChatGPT API Error: Timeout überschritten]"
        except self.BadRequestError as e:
             # --- Keep existing detailed BadRequestError handling ---
             # This error *should* be less likely now with image URLs, but keep the handling
             error_message = "N/A"
             error_body = getattr(e, 'response', None) # Get the response object
             status_code = getattr(e, 'status_code', 'N/A')
             if error_body:
                 try:
                     # Try to parse the JSON body for details
                     error_details = error_body.json()
                     error_message = error_details.get('error', {}).get('message', str(e))
                 except Exception: # Fallback if body is not JSON or parsing fails
                     error_message = getattr(e, 'message', str(e)) # Use the exception's message
             else: # Fallback if no response object
                 error_message = getattr(e, 'message', str(e))

             print(f"Fehler [ChatGPT]: BadRequestError Status {status_code}: {error_message}")

             if status_code == 429: # Rate limit
                  return f"[ChatGPT API Error: Rate Limit Exceeded (429) - {error_message}]"
             # Check for specific content filtering or invalid URL errors
             if 'content_policy_violation' in error_message.lower() or (hasattr(e,'code') and e.code == 'content_policy_violation'):
                  return f"[ChatGPT API Error: Content Policy Violation ({status_code}) - {error_message}]"
             if 'url' in error_message.lower() or 'fetch' in error_message.lower() or 'download' in error_message.lower():
                  return f"[ChatGPT API Error: Konnte Frame-URL nicht verarbeiten ({status_code}) - Prüfe URL-Gültigkeit/Zugriff. Details: {error_message}]"
             elif 'Invalid MIME type' in error_message or 'invalid_image_format' in str(getattr(e,'code', '')):
                  # This *shouldn't* happen with JPGs, but good to keep
                  return f"[ChatGPT API Error: Ungültiger MIME-Typ/Format ({status_code}) - Problem mit Frame-Upload/URL? Details: {error_message}]"
             else:
                  return f"[ChatGPT API Error: BadRequest {status_code} - {error_message}]"

        except self.APIError as e: # Catch other API errors (e.g., 5xx server errors)
            status_code = getattr(e, 'status_code', 'N/A')
            message = getattr(e, 'message', str(e))
            print(f"Fehler [ChatGPT]: APIError Status {status_code}: {message}")
            return f"[ChatGPT API Error: {status_code} - {message}]"
        except Exception as e: # Catch any other unexpected errors
            print(f"Fehler [ChatGPT]: Allgemeiner Fehler beim API-Aufruf: {type(e).__name__} - {e}")
            import traceback
            print(traceback.format_exc())
            return f"[ChatGPT API Error: Unexpected {type(e).__name__}]"


# --- Hilfsfunktionen ---

def read_files(filepaths: Optional[List[str]]) -> Optional[str]:
    """
    Liest den Inhalt mehrerer TEXT-Dateien und kombiniert sie.

    Args:
        filepaths: Eine Liste von Dateipfaden oder None.

    Returns:
        Ein String mit dem kombinierten Inhalt aller erfolgreich
        gelesenen Textdateien, oder None, wenn keine Dateien
        angegeben wurden oder keine gelesen werden konnten.
    """
    all_content = []
    if not filepaths:
        print("Info [read_files]: Keine Dateipfade angegeben.")
        return None # Return None if the input list is empty or None

    print(f"Info [read_files]: Versuche, Text-Kontext aus {len(filepaths)} Datei(en) zu lesen...")
    for filepath in filepaths:
        if not isinstance(filepath, str) or not filepath.strip():
            print(f"Warnung [read_files]: Ungültiger Dateipfad übersprungen: {filepath}")
            continue # Skip invalid entries

        resolved_path = os.path.expanduser(filepath) # Expand ~ if used

        try:
            # Check if the file exists and is a file
            if not os.path.isfile(resolved_path):
                print(f"Warnung [read_files]: Datei nicht gefunden oder ist kein regulärer File: {resolved_path}")
                continue # Skip directories or non-existent files

            # Attempt to read as UTF-8 text
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Add a header indicating the source file
                file_basename = os.path.basename(resolved_path)
                all_content.append(f"--- Inhalt von {file_basename} ---\n{content}")
                print(f"Info [read_files]: Textinhalt aus '{resolved_path}' erfolgreich gelesen.")
        # Specific exception for file not found (redundant with isfile check, but good practice)
        except FileNotFoundError:
            print(f"Warnung [read_files]: Datei nicht gefunden (erneute Prüfung): {resolved_path}")
        # Handle permission errors
        except PermissionError:
             print(f"Warnung [read_files]: Keine Leseberechtigung für Datei: {resolved_path}")
        # Handle encoding errors
        except UnicodeDecodeError:
            print(f"Warnung [read_files]: Datei '{resolved_path}' konnte nicht als UTF-8 Text gelesen werden (vermutlich Binärdatei oder falsche Kodierung, ignoriert).")
        # Catch other potential OS errors during file access
        except OSError as e:
             print(f"Warnung [read_files]: OS-Fehler beim Lesen der Datei {resolved_path}: {e}")
        # Catch any other unexpected errors
        except Exception as e:
            print(f"Warnung [read_files]: Allgemeiner Fehler beim Lesen der Datei {resolved_path}: {type(e).__name__} - {e}")

    if not all_content:
        print("Info [read_files]: Kein Text-Kontext aus Dateien geladen.")
        return None # Return None if no files were successfully read

    print("Info [read_files]: Text-Kontext erfolgreich aus Dateien kombiniert.")
    # Join the content from all files with double newlines
    return "\n\n".join(all_content)

# --- Beispielaufruf (zum Testen) ---
if __name__ == '__main__':
    # Erstelle temporäre Testdateien
    with open("test1.txt", "w", encoding="utf-8") as f:
        f.write("Dies ist Inhalt von Test 1.\nZeile 2.")
    with open("test2.log", "w", encoding="utf-8") as f:
        f.write("Log-Eintrag A\nLog-Eintrag B")
    with open("binary_test.bin", "wb") as f:
        f.write(b'\x00\x01\x02\x03') # Eine Binärdatei

    test_files = ["test1.txt", "non_existent.txt", "test2.log", "binary_test.bin", "/etc/hosts"] # Beispiel mit existierenden, nicht-existierenden, binären und Systemdateien

    combined_text = read_files(test_files)

    if combined_text:
        print("\n--- Kombinierter Text ---")
        print(combined_text)
    else:
        print("\nKein Text konnte aus den angegebenen Dateien gelesen werden.")

    # Aufräumen
    try:
        os.remove("test1.txt")
        os.remove("test2.log")
        os.remove("binary_test.bin")
    except OSError:
        pass

      
def get_windowed_history(full_history: List[Dict[str, str]], window_size: int) -> List[Dict[str, str]]:
    """Gibt die letzten 'window_size' Nachrichten aus dem Verlauf zurück."""
    if window_size <= 0:
        return []
    # Ensure we don't exceed the actual history length
    start_index = max(0, len(full_history) - window_size)
    return full_history[start_index:] # Return the slice from start_index to the end

    


def extract_relevant_frames(video_path: str, threshold_perc: float = 0.7, min_delta_thresh: int = 5, max_frames: Optional[int] = None) -> List[str]:
    """
    Extrahiert Frames aus einem Video, wenn signifikante Änderungen erkannt werden.

    Args:
        video_path: Pfad zur Videodatei.
        threshold_perc: Prozentualer Anteil der Pixel, die sich ändern müssen (0-100).
        min_delta_thresh: Mindest-Intensitätsänderung pro Pixel, die berücksichtigt wird.
        max_frames: Optionale maximale Anzahl an zu extrahierenden Frames.

    Returns:
        Liste mit Pfaden zu den gespeicherten relevanten Frame-Dateien (.jpg).
        Gibt leere Liste zurück bei Fehlern oder wenn keine Frames extrahiert wurden.
    """
    if not cv2:
        print("Fehler [Frame Extraction]: OpenCV (cv2) nicht verfügbar.")
        return []
    if not np:
        print("Fehler [Frame Extraction]: NumPy (np) nicht verfügbar.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler [Frame Extraction]: Video konnte nicht geöffnet werden: {video_path}")
        return []

    prev_frame_gray = None
    frame_id = 0
    extracted_count = 0
    relevant_frames = []

    # Create output directory relative to video, handle potential errors
    try:
        output_dir = os.path.join(os.path.dirname(video_path) or '.', f"extracted_frames_{os.path.splitext(os.path.basename(video_path))[0]}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Info [Frame Extraction]: Speichere Frames in: {output_dir}")
    except OSError as e:
        print(f"Fehler [Frame Extraction]: Konnte Output-Verzeichnis nicht erstellen: {output_dir} - {e}")
        cap.release()
        return []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Performance: Process every Nth frame? Maybe not needed if diff check is efficient
        # if frame_id % 2 != 0: # Example: process every 2nd frame
        #     frame_id += 1
        #     continue

        try:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0) # Blur more for robustness
        except cv2.error as e:
            print(f"Warnung [Frame Extraction]: Fehler bei Frame {frame_id} Konvertierung/Blur: {e}")
            frame_id += 1
            continue # Skip problematic frame


        significant_change = False
        if prev_frame_gray is not None:
            # Calculate difference
            frame_delta = cv2.absdiff(prev_frame_gray, current_frame_gray)
            # Apply threshold to get pixels that changed significantly
            thresh = cv2.threshold(frame_delta, min_delta_thresh, 255, cv2.THRESH_BINARY)[1]
            # Count non-zero pixels (changed pixels)
            non_zero_count = np.count_nonzero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            change_percentage = (non_zero_count / total_pixels) * 100

            # Check if change exceeds threshold percentage
            if change_percentage > threshold_perc:
                significant_change = True
                # print(f"Debug: Frame {frame_id}, Change: {change_percentage:.2f}% > {threshold_perc}%") # Debug output

        # Always save the very first frame, or frames with significant change
        if frame_id == 0 or significant_change:
            if max_frames is not None and extracted_count >= max_frames:
                print(f"Info [Frame Extraction]: Maximale Anzahl ({max_frames}) an Frames erreicht.")
                break # Stop extracting if limit is reached

            frame_filename = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg") # Use more digits for longer videos
            try:
                success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # Save as JPG with quality setting
                if success:
                    relevant_frames.append(frame_filename)
                    extracted_count += 1
                    # print(f"Info [Frame Extraction]: Relevanten Frame gespeichert: {frame_filename}") # Verbose
                else:
                     print(f"Warnung [Frame Extraction]: Konnte Frame nicht speichern: {frame_filename}")
            except cv2.error as e:
                 print(f"Warnung [Frame Extraction]: Fehler beim Speichern von Frame {frame_id}: {e}")
            except Exception as e: # Catch other potential errors during save
                 print(f"Warnung [Frame Extraction]: Unerwarteter Fehler beim Speichern von Frame {frame_id}: {e}")

        prev_frame_gray = current_frame_gray
        frame_id += 1

    cap.release()
    print(f"Info [Frame Extraction]: Extraktion beendet. {len(relevant_frames)} Frames gespeichert.")
    return relevant_frames


# Keep upload_to_gcs_and_get_public_url as it is (it works for images too)
def upload_to_gcs_and_get_public_url(local_file_path: str, bucket_name: str, project_id: Optional[str] = None, destination_blob_name: Optional[str] = None) -> Optional[str]:
    """
    Lädt eine Datei zu Google Cloud Storage hoch und gibt die öffentliche URL zurück.
    Verlässt sich auf Bucket IAM (Uniform Access) für öffentliche Lesbarkeit.
    """
    # (Keep existing implementation, it should work fine for JPG frames)
    if not storage:
        print("Fehler [GCS]: google-cloud-storage Bibliothek nicht geladen.")
        return None

    if not os.path.exists(local_file_path):
        print(f"Fehler [GCS]: Lokale Datei nicht gefunden: '{local_file_path}'")
        return None

    try:
        # Initialisiere GCS Client mit expliziter Projekt ID
        storage_client = None # Initialize to None
        if project_id:
            storage_client = storage.Client(project=project_id)
            # print(f"Info [GCS]: Initialisiere Client für Projekt '{project_id}'.") # Less verbose now
        else:
            # Fallback (könnte fehlschlagen, wie zuvor gesehen)
            storage_client = storage.Client()
            print("Warnung [GCS]: Projekt ID nicht explizit angegeben. Client versucht, sie abzuleiten.")

        # Hole Bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except NotFound:
            print(f"Fehler [GCS]: Bucket '{bucket_name}' nicht gefunden.")
            return None
        except Forbidden:
             print(f"Fehler [GCS]: Zugriff auf Bucket '{bucket_name}' verweigert. Prüfe IAM-Berechtigungen.")
             return None
        except AttributeError:
             print(f"Fehler [GCS]: Storage Client konnte nicht korrekt initialisiert werden (wahrscheinlich fehlende Projekt-ID oder Credentials).")
             return None


        # Bestimme Zieldateiname in GCS
        if destination_blob_name is None:
            destination_blob_name = os.path.basename(local_file_path)

        # Erstelle Blob-Objekt
        blob = bucket.blob(destination_blob_name)

        # Errate MIME-Typ
        mime_type, _ = mimetypes.guess_type(local_file_path)
        if mime_type is None:
            # Default for frames should be image/jpeg if extraction uses .jpg
            mime_type = 'image/jpeg' if destination_blob_name.lower().endswith(".jpg") else 'application/octet-stream'
            print(f"Warnung [GCS]: Konnte MIME-Typ für {destination_blob_name} nicht raten. Verwende {mime_type}.")

        # Lade Datei hoch
        # print(f"Info [GCS]: Lade '{local_file_path}' hoch nach 'gs://{bucket_name}/{destination_blob_name}'...") # Less verbose
        blob.upload_from_filename(local_file_path, content_type=mime_type)
        # print("Info [GCS]: Upload abgeschlossen.") # Less verbose

        # Konstruiere öffentliche URL manuell (URL-Encoding für Dateinamen)
        public_url = f"https://storage.googleapis.com/{bucket_name}/{quote(destination_blob_name)}" # Ensure quote is imported

        # print(f"Erfolg [GCS]: Öffentliche URL: {public_url}") # Less verbose
        return public_url

    except Forbidden as e:
        print(f"Fehler [GCS]: Zugriff verweigert. Prüfe Credentials und IAM Rollen (z.B. 'Storage Object Admin' für Upload). Details: {e}")
        return None
    except OSError as e: # Catch project ID error more specifically
        if "Project was not passed" in str(e) or "Could not determine project ID" in str(e):
             print(f"Fehler [GCS]: Projekt-ID konnte nicht ermittelt werden: {e}. Stelle sicher, dass GOOGLE_CLOUD_PROJECT gesetzt ist oder project_id übergeben wird.")
        else:
            print(f"Ein OS Fehler trat beim GCS Upload auf: {type(e).__name__} - {e}")
            # import traceback
            # print(traceback.format_exc()) # Optionally add traceback
        return None
    except Exception as e:
        print(f"Ein unerwarteter Fehler trat beim GCS Upload auf: {type(e).__name__} - {e}")
        # import traceback
        # print(traceback.format_exc()) # Optionally add traceback
        return None