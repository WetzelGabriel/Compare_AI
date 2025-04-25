# Compare_AI_CHT.py (oder streamlit_app.py)
# Korrigierter Syntaxfehler in cleanup_temp_data

import streamlit as st
import os
import tempfile
import threading
import time
import numpy as np
import shutil
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
import traceback

# Importiere die Bibliothek (Version, die Strings zurückgibt)
import LIB_LLMChatLib as myllm

def myCompare_CHT_App():

    # ****** 1. Environment laden (einmalig) ******
    dotenv_path = '/home/gabriel/myProject/myvenv/.env' # <<< PRÜFE DIESEN PFAD GENAU!
    if os.path.exists(dotenv_path):
        print(f"Lade Umgebungsvariablen aus {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        print(f"Warnung: .env-Datei nicht gefunden unter {dotenv_path}.")

    

    # Umgebungsvariablen lesen
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    # Standardwerte
    DEFAULT_HISTORY_SIZE = 10
    DEFAULT_MAX_FRAMES_GPT = 10

    # --- Thread-sicherer Logger ---
    _log_lock = threading.Lock()
    def add_system_log(message: str):
        with _log_lock:
            if "system_log" not in st.session_state: st.session_state.system_log = []
            log_entry = f"{time.strftime('%H:%M:%S')} - {message}"
            st.session_state.system_log.append(log_entry)
            st.session_state.system_log = st.session_state.system_log[-200:]

    # --- Session State Initialisierung ---
    def initialize_session_state():
        defaults = {
            "system_log": ["System-Log initialisiert."],
            "chat_history_gemini": [],
            "chat_history_chatgpt": [],
            "current_video_processed": False,
            "uploaded_file_info": None,
            "gcs_video_url": None,
            "chatgpt_frame_urls": [],
            "processing_initial_analysis": False,
            "processing_follow_up": False,
            "trigger_initial_analysis": False,
            "llm_map": {},
            "llm_map_initialized": False,
            "last_user_prompt": None,
            "thread_results": {},
            "trigger_follow_up_analysis": False,
            "last_user_prompt_processed": None
        }
        for key, default_value in defaults.items():
            if key not in st.session_state: st.session_state[key] = default_value

    initialize_session_state()

    # --- LLM Instanzen ---
    @st.cache_resource
    def initialize_llms():
        llms = {}
        add_system_log("Initialisiere LLM-Clients...")
        initialization_errors = False
        if GEMINI_API_KEY:
            try:
                # Die Bibliothek selbst prüft den Import und gibt WARNUNG/Fehler aus
                llms["Gemini"] = myllm.GeminiLLM(api_key=GEMINI_API_KEY) # Kein Logger mehr übergeben
            except ImportError:
                add_system_log("FEHLER: google.generativeai nicht gefunden! Umgebung prüfen.")
                initialization_errors = True
            except Exception as e:
                add_system_log(f"Fehler Init Gemini: {e}")
                initialization_errors = True
        else: add_system_log("Info: Kein GEMINI_API_KEY.")

        if OPENAI_API_KEY:
            try:
                llms["ChatGPT"] = myllm.ChatGPTLLM(api_key=OPENAI_API_KEY) # Kein Logger mehr übergeben
            except Exception as e:
                add_system_log(f"Fehler Init ChatGPT: {e}")
                initialization_errors = True
        else: add_system_log("Info: Kein OPENAI_API_KEY.")

        if not llms: add_system_log("FEHLER: Keine LLMs initialisiert."); return None
        else: add_system_log(f"LLMs initialisiert: {list(llms.keys())}"); return llms


    if not st.session_state.llm_map_initialized:
        st.session_state.llm_map = initialize_llms()
        st.session_state.llm_map_initialized = True

    # --- Funktion zum Aufräumen ---
    def cleanup_temp_data(info_dict: Optional[Dict]):
        """Versucht, temporäre Dateien und Verzeichnisse zu löschen."""
        if not info_dict: return
        temp_path = info_dict.get("temp_path")
        frame_dir = info_dict.get("frame_dir")
        upload_dir = info_dict.get("upload_dir") # Verzeichnis, in dem Temp-Video liegt

        if temp_path and os.path.exists(temp_path):
            try: # <-- Korrigierte Einrückung
                os.remove(temp_path)
                add_system_log(f"Temp-Video gelöscht: {temp_path}")
            except Exception as e: # <-- Korrigierte Einrückung
                add_system_log(f"Warnung Löschen Temp-Video: {e}")

        if frame_dir and os.path.exists(frame_dir) and os.path.isdir(frame_dir):
            try: # <-- Korrigierte Einrückung
                shutil.rmtree(frame_dir) # Löscht Verzeichnis und Inhalt
                add_system_log(f"Temp-Frames gelöscht: {frame_dir}")
            except Exception as e: # <-- Korrigierte Einrückung
                add_system_log(f"Warnung Löschen Temp-Frames: {e}")

        # Lösche auch das Upload-Verzeichnis, falls es existiert
        if upload_dir and os.path.exists(upload_dir) and os.path.isdir(upload_dir):
            try: # <-- Korrigierte Einrückung
                shutil.rmtree(upload_dir)
                add_system_log(f"Upload-Verzeichnis gelöscht: {upload_dir}")
            except Exception as e: # <-- Korrigierte Einrückung
                add_system_log(f"Warnung Löschen Upload-Dir: {e}")

    # ==============================================================================
    # Streamlit UI Aufbau
    # ==============================================================================
    st.title("🎬 Video-Analyse Vergleich")

    if not st.session_state.llm_map:
        st.error("FEHLER: LLM-Clients nicht initialisiert. App kann nicht starten. Prüfe API-Keys/Umgebung.")
        # (Log anzeigen wie zuvor)
        with st.expander("System-Log / Meldungen", expanded=True):
            log_container = st.container(height=200)
            log_text = "\n".join(reversed(st.session_state.get("system_log", ["Keine Logs vorhanden."])))
            log_container.text_area("Logs:", value=log_text, height=180, disabled=True, key="log_display_area_error")
        st.stop()

    # --- Bereich 1: Datei-Upload ---
    uploaded_file = st.file_uploader(
        "Wähle eine Videodatei", type=["mp4", "avi", "mov", "mkv"], key="file_uploader"
    )

    # --- Thread Zielfunktion ---
    def run_llm_chat_thread(
        model_key: str,
        llm_instance: myllm.LLMInterface,
        prompt: str,
        history: List[Dict[str, str]],
        video_data: Optional[Dict],
        output_dict: Dict[str, Any]
    ):
        # (Implementation wie zuvor - gibt String zurück)
        try:
            response_str = llm_instance.chat(prompt, history, video_data=video_data)
            output_dict[model_key] = response_str
        except Exception as thread_e:
            print(f"FEHLER im {model_key} Thread: {thread_e}\n{traceback.format_exc()}")
            output_dict[model_key] = f"[Thread Error {model_key}: {type(thread_e).__name__} - {thread_e}]"

    # --- Bereich 2: Verarbeitung und Chat ---
    if uploaded_file is not None:
        new_upload = (st.session_state.uploaded_file_info is None or
                    st.session_state.uploaded_file_info["name"] != uploaded_file.name)

        if new_upload:
            # (Reset und Temp-Datei speichern wie zuvor)
            add_system_log(f"Neue Datei: {uploaded_file.name}. Reset.")
            cleanup_temp_data(st.session_state.uploaded_file_info)
            # State zurücksetzen (wichtige Teile)
            st.session_state.chat_history_gemini = []
            st.session_state.chat_history_chatgpt = []
            st.session_state.current_video_processed = False
            st.session_state.gcs_video_url = None
            st.session_state.chatgpt_frame_urls = []
            st.session_state.processing_initial_analysis = False
            st.session_state.processing_follow_up = False
            st.session_state.last_user_prompt = None
            st.session_state.thread_results = {}
            st.session_state.trigger_follow_up_analysis = False
            st.session_state.last_user_prompt_processed = None
            st.session_state.uploaded_file_info = None # Wird gleich neu gesetzt
            try:
                upload_dir = tempfile.mkdtemp(prefix="streamlit_uploads_")
                temp_path = os.path.join(upload_dir, uploaded_file.name)
                with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_info = {"name": uploaded_file.name, "temp_path": temp_path, "upload_dir": upload_dir, "frame_dir": None}
                add_system_log(f"Temp-Datei: {temp_path}")
                st.session_state.trigger_initial_analysis = True
            except Exception as e:
                st.error(f"Fehler beim Speichern der Datei: {e}")
                add_system_log(f"FEHLER Speichern Temp-Datei: {e}")
                st.session_state.uploaded_file_info = None

        # --- Start der initialen Analyse ---
        if st.session_state.get("trigger_initial_analysis", False) and not st.session_state.processing_initial_analysis:
            if not st.session_state.uploaded_file_info or not st.session_state.uploaded_file_info.get("temp_path"):
                st.error("Interner Fehler: Keine Temp-Datei für Analyse.")
                add_system_log("Fehler: Analyse Trigger ohne Temp-Datei.")
                st.session_state.trigger_initial_analysis = False
            else:
                st.session_state.processing_initial_analysis = True
                st.session_state.trigger_initial_analysis = False
                current_thread_results = {}

                with st.spinner(f"Verarbeite Video '{st.session_state.uploaded_file_info['name']}'..."):
                    temp_path = st.session_state.uploaded_file_info["temp_path"]
                    # --- Vorverarbeitung (wie zuvor) ---
                    add_system_log("Starte GCS Upload...")
                    gcs_url = None
                    if GOOGLE_CLOUD_PROJECT_ID and GCS_BUCKET_NAME:
                        try: gcs_url = myllm.upload_to_gcs_and_get_public_url(temp_path, GCS_BUCKET_NAME, GOOGLE_CLOUD_PROJECT_ID)
                        except Exception as e: add_system_log(f"FEHLER GCS Upload: {e}")
                    if gcs_url: st.session_state.gcs_video_url = gcs_url; add_system_log(f"GCS OK: {gcs_url}")
                    else: add_system_log("Warnung GCS Upload: None/Fehler"); st.session_state.gcs_video_url = None

                    add_system_log("Starte Frame-Extraktion...")
                    extracted_frames = []
                    temp_frame_dir = None
                    try:
                        extracted_frames = myllm.extract_relevant_frames(temp_path, max_frames=DEFAULT_MAX_FRAMES_GPT * 3)
                        if extracted_frames: temp_frame_dir = os.path.dirname(extracted_frames[0]); st.session_state.uploaded_file_info["frame_dir"] = temp_frame_dir; add_system_log(f"{len(extracted_frames)} Frames extrahiert: {temp_frame_dir}")
                        else: add_system_log("Info: Keine Frames extrahiert.")
                    except Exception as e: add_system_log(f"FEHLER Frame-Extraktion: {e}")

                    frame_urls = []
                    if extracted_frames and temp_frame_dir:
                        frames_to_upload = extracted_frames
                        if len(extracted_frames) > DEFAULT_MAX_FRAMES_GPT:
                            add_system_log(f"Reduziere Frames: {len(extracted_frames)} -> {DEFAULT_MAX_FRAMES_GPT}.")
                            indices = np.linspace(0, len(extracted_frames) - 1, DEFAULT_MAX_FRAMES_GPT, dtype=int)
                            frames_to_upload = [extracted_frames[i] for i in indices]
                        add_system_log(f"Starte GCS Upload für {len(frames_to_upload)} Frames...")
                        video_base_name = os.path.splitext(os.path.basename(temp_path))[0].replace(" ", "_")
                        gcs_frame_dir = f"extracted_frames_streamlit/{video_base_name}_{int(time.time())}"
                        if GOOGLE_CLOUD_PROJECT_ID and GCS_BUCKET_NAME:
                            for i, frame_path in enumerate(frames_to_upload):
                                try:
                                    f_url = myllm.upload_to_gcs_and_get_public_url(frame_path, GCS_BUCKET_NAME, GOOGLE_CLOUD_PROJECT_ID, f"{gcs_frame_dir}/{os.path.basename(frame_path)}")
                                    if f_url: frame_urls.append(f_url)
                                    else: add_system_log(f"Warnung Frame Upload (None): {frame_path}")
                                except Exception as e: add_system_log(f"FEHLER Frame Upload: {frame_path} - {e}")
                            add_system_log(f"GCS Frame Upload OK: {len(frame_urls)}/{len(frames_to_upload)}.")
                        else: add_system_log("Info: Frame Upload übersprungen.")
                    st.session_state.chatgpt_frame_urls = frame_urls
                    # --- Ende Vorverarbeitung ---

                    # --- Initiale LLM-Anfrage ---
                    initial_prompt_detail = f"(Video: {st.session_state.uploaded_file_info['name']})"
                    if st.session_state.chatgpt_frame_urls:
                        initial_prompt_detail += f". {len(st.session_state.chatgpt_frame_urls)} Bilder für ChatGPT."
                    initial_user_input = f"""
    AI Video Analyse Prompt: Kritische Sicherheitsereignisse
    Ziel: Analysiere das bereitgestellte Sicherheitskamera-Videomaterial {initial_prompt_detail}.
    **Für ChatGPT:** Bitte analysiere die Bilder, die als URLs bereitgestellt werden, sorgfältig.
    Suche ausschließlich auf Anzeichen von Einbruch/Gewaltsames Eindringen und Feuer. Generiere prägnante, sofort verwertbare Alarme NUR für diese Ereignisse.
    Eingabe: Video ({st.session_state.uploaded_file_info['name']}) direkt für Gemini oder extrahierte Frames (als Bild-URLs für ChatGPT).
    Zu detektierende kritische Trigger: Einbruch/Gewaltsames Eindringen, Feuer.
    Ausgabeformat (NUR bei erkanntem Trigger): KRITISCHER ALARM...
    Anweisungen: Ignoriere Normalaktivität... Wenn nichts erkannt: NUR 'STATUS: Keine kritischen Ereignisse detektiert.'... Video-Beschreibung hinzufügen...
    """
                    add_system_log("Sende initiale Analyseanfrage an LLMs...")
                    gemini_video_input = {"video_path": st.session_state.uploaded_file_info["temp_path"]}
                    chatgpt_video_input = {"frame_urls": st.session_state.chatgpt_frame_urls if st.session_state.chatgpt_frame_urls else None}

                    threads = []
                    llm_map = st.session_state.llm_map
                    if "Gemini" in llm_map:
                        thread_gemini = threading.Thread(target=run_llm_chat_thread, args=("Gemini", llm_map["Gemini"], initial_user_input, [], gemini_video_input, current_thread_results), daemon=True, name="GeminiThread")
                        threads.append(thread_gemini)
                        thread_gemini.start()
                    if "ChatGPT" in llm_map:
                        thread_chatgpt = threading.Thread(target=run_llm_chat_thread, args=("ChatGPT", llm_map["ChatGPT"], initial_user_input, [], chatgpt_video_input, current_thread_results), daemon=True, name="ChatGPTThread")
                        threads.append(thread_chatgpt)
                        thread_chatgpt.start()

                    # Warten auf Threads
                    max_wait = 700; start_wait = time.time(); timed_out = False
                    while any(t.is_alive() for t in threads):
                        if (time.time() - start_wait) > max_wait: timed_out = True; break
                        time.sleep(0.5)
                    for t in threads: # Fehler für Timeout speichern
                        if t.is_alive():
                            add_system_log(f"Warnung: Thread {t.name} Timeout ({max_wait}s).")
                            model_key = t.name.replace("Thread", "")
                            if model_key not in current_thread_results: current_thread_results[model_key] = f"[{model_key} FEHLER: Timeout]"

                    # Ergebnisse SOFORT verarbeiten
                    add_system_log(f"Verarbeite initiale Thread-Ergebnisse: {list(current_thread_results.keys())}")
                    gemini_result_str = current_thread_results.get("Gemini")
                    chatgpt_result_str = current_thread_results.get("ChatGPT")
                    if gemini_result_str: st.session_state.chat_history_gemini.append({"role": "model", "content": str(gemini_result_str)})
                    if chatgpt_result_str: st.session_state.chat_history_chatgpt.append({"role": "assistant", "content": str(chatgpt_result_str)})

                    if not timed_out: add_system_log("Initiale Analyse abgeschlossen.")
                    st.session_state.current_video_processed = True
                    st.session_state.processing_initial_analysis = False
                    # Kein Rerun mehr nötig hier

                # Ende des 'with st.spinner'

        # --- Chat-Anzeige ---
        if st.session_state.current_video_processed:
            st.divider()
            st.subheader("Chat Verlauf")
            col1, col2 = st.columns(2)
            def display_chat(column, history_key, model_name):
                with column:
                    st.markdown(f"**{model_name}**")
                    chat_container = st.container(height=400)
                    for msg in st.session_state[history_key]:
                        with chat_container.chat_message(msg["role"]):
                            st.write(msg["content"])
            display_chat(col1, "chat_history_gemini", "Gemini")
            display_chat(col2, "chat_history_chatgpt", "ChatGPT")

        # --- Benutzereingabe für Folgefragen ---
        input_disabled = st.session_state.processing_initial_analysis or st.session_state.processing_follow_up

        user_prompt = st.chat_input(
            "Stelle eine Frage zum Video...",
            key="chat_input_widget",
            disabled=input_disabled,
        )

        if user_prompt and not input_disabled:
            st.session_state.processing_follow_up = True # Verarbeitung startet
            add_system_log(f"Verarbeite Folgeanfrage: {user_prompt}")
            # Füge User-Prompt SOFORT zur History hinzu (wird beim nächsten Rerun angezeigt)
            st.session_state.chat_history_gemini.append({"role": "user", "content": user_prompt})
            st.session_state.chat_history_chatgpt.append({"role": "user", "content": user_prompt})

            # Verlauf für API vorbereiten
            def get_text_history(history_list):
                # Nimm alle bis auf den *gerade hinzugefügten* User-Prompt
                return [{"role": m["role"], "content": m["content"]} for m in history_list[:-1]]

            hist_gemini_api = myllm.get_windowed_history(get_text_history(st.session_state.chat_history_gemini), DEFAULT_HISTORY_SIZE)
            hist_chatgpt_api = myllm.get_windowed_history(get_text_history(st.session_state.chat_history_chatgpt), DEFAULT_HISTORY_SIZE)

            # ====> ANPASSUNG HIER <====
            # Für Gemini: KEIN Video erneut senden (zu ineffizient)
            gemini_follow_up_data = None
            # Für ChatGPT: Frame URLs ERNEUT senden, um visuellen Kontext zu behalten
            chatgpt_follow_up_data = {"frame_urls": st.session_state.chatgpt_frame_urls if st.session_state.chatgpt_frame_urls else None}
            if chatgpt_follow_up_data["frame_urls"]:
                add_system_log(f"Sende {len(st.session_state.chatgpt_frame_urls)} Frame URLs erneut an ChatGPT für Kontext.")
            else:
                add_system_log("Keine Frame URLs vorhanden für ChatGPT-Kontext.")
            # ====> ENDE ANPASSUNG <====

            add_system_log("Sende Folgeanfrage an LLMs...")
            current_thread_results = {} # Lokales Dict für diese Runde

            # Threads starten (run_llm_chat_thread Funktion bleibt unverändert)
            threads = []
            llm_map = st.session_state.llm_map
            if "Gemini" in llm_map:
                # Übergibt gemini_follow_up_data (ist None)
                thread_gemini = threading.Thread(target=run_llm_chat_thread, args=("Gemini", llm_map["Gemini"], user_prompt, hist_gemini_api, gemini_follow_up_data, current_thread_results), daemon=True, name="GeminiThread")
                threads.append(thread_gemini)
                thread_gemini.start()
            if "ChatGPT" in llm_map:
                # Übergibt chatgpt_follow_up_data (enthält URLs)
                thread_chatgpt = threading.Thread(target=run_llm_chat_thread, args=("ChatGPT", llm_map["ChatGPT"], user_prompt, hist_chatgpt_api, chatgpt_follow_up_data, current_thread_results), daemon=True, name="ChatGPTThread")
                threads.append(thread_chatgpt)
                thread_chatgpt.start()

            # WARTE auf Threads
            max_wait = 700; start_wait = time.time(); timed_out = False
            with st.spinner("Warte auf LLM-Antworten..."):
                while any(t.is_alive() for t in threads):
                    if (time.time() - start_wait) > max_wait: timed_out = True; break
                    time.sleep(0.5)
            for t in threads: # Fehler für Timeout speichern
                if t.is_alive():
                    add_system_log(f"Warnung: Thread {t.name} Timeout ({max_wait}s).")
                    model_key = t.name.replace("Thread", "")
                    if model_key not in current_thread_results: current_thread_results[model_key] = f"[{model_key} FEHLER: Timeout]"

            # Ergebnisse SOFORT verarbeiten
            add_system_log(f"Verarbeite Folge-Thread-Ergebnisse: {list(current_thread_results.keys())}")
            gemini_result_str = current_thread_results.get("Gemini")
            chatgpt_result_str = current_thread_results.get("ChatGPT")
            # Füge Ergebnisse zur jeweiligen History hinzu
            if gemini_result_str: st.session_state.chat_history_gemini.append({"role": "model", "content": str(gemini_result_str)})
            if chatgpt_result_str: st.session_state.chat_history_chatgpt.append({"role": "assistant", "content": str(chatgpt_result_str)})

            if not timed_out: add_system_log("Folgeanfrage abgeschlossen.")
            st.session_state.processing_follow_up = False
            # Rerun auslösen, um die neuen Nachrichten anzuzeigen
            st.rerun()


    # --- Bereich 3: System-Log ---
    st.divider()
    with st.expander("System-Log / Meldungen", expanded=False):
        log_container = st.container(height=200)
        log_text = "\n".join(reversed(st.session_state.get("system_log", ["Keine Logs vorhanden."])))
        log_container.text_area("Logs:", value=log_text, height=180, disabled=True, key="log_display_area")

if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    # Streamlit app title
    # ****** 2. Page Config (als erster st-Befehl) ******
    st.set_page_config(
        layout="wide",
        page_title="Video Analyse Vergleich",
        page_icon="🎬"
    )
    myCompare_CHT_App()