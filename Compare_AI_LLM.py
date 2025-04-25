#####################################################
# Programm for llm abd further model comparison     #
# run it with 'streamlit run Compare_AI_LLM.py'     #
#####################################################
# 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time
import gc

def myCompare_LLM_App():
    
    # Input field for the question
    question = st.text_input("Enter your question here")

    # Template for the question
    template = """Question: {question}

    Answer: """

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Function to get model response
    def get_model_response(model_name, question):
        model = OllamaLLM(model=model_name)
        response = prompt | model
        del model  # Delete the model after use
        gc.collect()  # Explicitly call garbage collector
        return response.invoke({"question": question})

    # If a question is entered
    if question:
        # Create three columns
        col1, col2, col3 = st.columns([1, 1, 1])

        # Calculate responses and times for each model
        with col1:
            st.subheader("Dolphin3 GGUF")
            start_time = time.time()
            response = get_model_response("hf.co/bartowski/Dolphin3.0-Llama3.2-1B-GGUF:Q6_K_L", question)
            end_time = time.time()
            duration = end_time - start_time
            st.write(f"Time taken: {duration:.2f} seconds")
            st.write(response)

        with col2:
            st.subheader("DeepSeek-R1 1.5b")
            start_time = time.time()
            response = get_model_response("deepseek-r1:1.5b", question)
            end_time = time.time()
            duration = end_time - start_time
            st.write(f"Time taken: {duration:.2f} seconds")
            st.write(response)

        with col3:
            st.subheader("Gemini Qwen2.5 0.5B")
            start_time = time.time()
            response = get_model_response("hf.co/mradermacher/Gemini-Distill-Qwen2.5-0.5B-ead-GGUF:Q8_0", question)
            end_time = time.time()
            duration = end_time - start_time
            st.write(f"Time taken: {duration:.2f} seconds")
            st.write(response)


if __name__ == "__main__":
    # If the script is run directly, execute the Streamlit app
    st.title("LLM Comparison App")
    myCompare_LLM_App()