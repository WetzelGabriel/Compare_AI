# LLM and AI Model Comparison App

This Streamlit app provides a **modular interface** for comparing various AI models including LLMs, Speech-to-Text, Video analysis, and more.

---

## 🚀 How to Run

Make sure your environment variables are set (via `.streamlit/secrets.toml`).

Then simply run:

```bash
streamlit run Compare_AI_Main.py
```

---

## 📜 Features

- **LLM Comparison:**  
  Compare outputs from different Large Language Models (LLMs).

- **Speech-To-Text Comparison:**  
  Benchmark various STT models for transcription quality.

- **Video-To-Text Application:**  
  Extract speech from video files and transcribe it.

- **Speech Recording Application:**  
  Record your voice and analyze results with models.

- **Security Video Analysis Application:**  
  Analyze video footage for security and surveillance use-cases.

- **Candidate Interview Application:**  
  Simulate and analyze AI-driven interview scenarios.

---

## 🔑 Environment Variables

This app requires a few environment variables, accessible from `st.secrets`:

| Key                  | Description                                  |
|-----------------------|----------------------------------------------|
| `OPENAI_API_KEY`       | OpenAI API Key                               |
| `GEMINI_API_KEY`       | Google Gemini API Key                        |
| `GCS_BUCKET_NAME`      | GCS (Google Cloud Storage) Bucket Name       |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud Project ID                     |

These should be set either in your **`.streamlit/secrets.toml`** or your deployment platform (e.g., Streamlit Cloud).

---

## 🖥 Interface Overview

- **Sidebar Menu:**  
  Choose the application module you'd like to interact with.

- **Main Panel:**  
  Displays outputs, comparisons, and app-specific interactions based on your selection.

---

## 🛠 Project Structure

- `Compare_AI_Main.py` — Main entry point and navigation.
- `Compare_AI_LLM.py` — LLM comparison app.
- `Compare_AI_STT.py` — Speech-to-Text comparison app.
- `Compare_AI_VTT.py` — Video-to-Text app.
- `Compare_AI_REC.py` — Audio recording app.
- `Compare_AI_CHT.py` — Security video analysis app.
- `Compare_AI_CV.py` — Candidate interview simulation.

---

## 📚 LIB Files

The **LIB files** are modular Python files that contain the actual implementations for each sub-application.  
Each LIB file focuses on a specific AI task and can be individually developed, extended, or replaced.

| File                  | Description                                              |
|------------------------|----------------------------------------------------------|
| `Compare_AI_LLM.py`    | Functions and UI for comparing large language models (LLMs). |
| `Compare_AI_STT.py`    | Functions and UI for speech-to-text model comparison.    |
| `Compare_AI_VTT.py`    | Functions to process video and extract speech as text.   |
| `Compare_AI_REC.py`    | Functions to record speech and evaluate models.          |
| `Compare_AI_CHT.py`    | Functions for analyzing security footage using AI.       |
| `Compare_AI_CV.py`     | Functions for candidate interviews using AI agents.      |

> ✨ **Tip:** You can easily add new LIB files following the same structure to expand the app with more comparison functionalities.

---
Perfect, here's what I'll do:

1. **Diagram** — showing `Compare_AI_Main.py` and the LIB files it imports.
2. **New Chapter** — about **Jupyter notebooks** for testing the same functionality.

---

# 📈 System Diagram

Here's a simple diagram showing how the main file connects to the LIB files:

```
Compare_AI_Main.py
   │
   ├── imports ───▶ Compare_AI_LLM.py     (LLM comparison functions)
   │
   ├── imports ───▶ Compare_AI_STT.py     (Speech-to-Text functions)
   │
   ├── imports ───▶ Compare_AI_VTT.py     (Video-to-Text functions)
   │
   ├── imports ───▶ Compare_AI_REC.py     (Speech recording functions)
   │
   ├── imports ───▶ Compare_AI_CHT.py     (Security video analysis functions)
   │
   └── imports ───▶ Compare_AI_CV.py      (Candidate interview functions)
```

The **`Compare_AI_Main.py`** acts like a **menu** and **navigation** interface that ties all the LIBs together in one place through the Streamlit UI.

---

# 📓 Jupyter Notebooks for Testing

In addition to running the application via Streamlit,  
you can also **test all functionalities individually** using **Jupyter notebooks** provided in the repository.

Each notebook corresponds to a particular module and uses the same underlying library (`Compare_AI_LLM.py`, `Compare_AI_STT.py`, etc.).

✅ **Advantages of using the notebooks:**
- Quickly test individual models without running the entire app.
- Easier to debug, tweak, or visualize intermediate outputs.
- Useful for iterative development and model benchmarking.

✅ **How to use:**
- Open the Jupyter Notebook (`.ipynb`) that matches the module you want to test.
- Run the cells to interact directly with the models.
- Notebooks automatically import the corresponding LIB files.

✅ **Location:**
- The notebook files are located directly inside the repository alongside the Python modules.

✅ **Requirements:**
- Make sure you have `jupyter` installed (`pip install notebook`) or use an IDE like **VSCode** or **JupyterLab** to open them.

✅ **In short:**  
- **Streamlit App** = Run full UI for users.  
- **Jupyter Notebooks** = Run individual functions for development and testing.

---

## 📬 Contact

Feel free to contribute, raise issues, or suggest new comparison modules!
