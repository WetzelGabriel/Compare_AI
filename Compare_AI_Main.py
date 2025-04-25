#####################################################
# Programm for llm abd further model comparison     #
# run it with 'streamlit run Compare_AI_Main.py'    #
#####################################################
import streamlit as st
from Compare_AI_LLM import myCompare_LLM_App
from Compare_AI_STT import myCompare_STT_App
from Compare_AI_VTT import myCompare_VTT_App
from Compare_AI_REC import myCompare_REC_App
from Compare_AI_CHT import myCompare_CHT_App
from Compare_AI_CV import myCompare_CV_App


# Streamlit interface
st.title("LLM and AI Model Comparison App")

# Define menu options and their corresponding actions
menu_actions = {
    "LLM Comparison": (myCompare_LLM_App, "You are in the LLM Comparison section."),
    "Speech-To-Text Comparison": (myCompare_STT_App, "You are in the Speech-To-Text Comparison section."),
    "Video-To-Speech Application": (myCompare_VTT_App, "You are in the Video-To-Text section."),
    "Speech Recording Application": (myCompare_REC_App, "You are in the Speech Recording section."),
    "Security Video-Analysis Application": (myCompare_CHT_App, "You are in the Security Video Analysis section."),
    "Candidate Interview": (myCompare_CV_App, "You are in the Candidate Interview section."),   
}

# Sidebar menu for subapplication selection
menu = st.sidebar.selectbox(
    "Select Subapplication",
    list(menu_actions.keys())  # Dynamically populate the menu options
)

# Display selected subapplication
if menu in menu_actions:
    action, message = menu_actions[menu]
    st.sidebar.write(message)
    if action:  # Check if the action is defined
        action()
else:
    st.sidebar.write("Invalid selection. Please choose a valid option.")