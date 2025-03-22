# Q&A Chatbot
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env

# Configure Google API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to load the Generative model and get responses
def get_gemini_response(input_text, image, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        response = model.generate_content([input_text, image[0], prompt])
        return response.text
    except Exception as e:
        return f"Error fetching response: {str(e)}"

# Function to set up the input image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

# User inputs
input_text = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_data = ""

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the image")

input_prompt = """
    You are an expert in understanding invoices.
    You will receive input images as invoices &
    you will have to answer questions based on the input image.
"""

# If the submit button is clicked
if submit:
    if not input_text.strip():
        st.warning("Please enter a prompt.")
    elif not uploaded_file:
        st.warning("Please upload an image.")
    else:
        try:
            image_parts = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt, image_parts, input_text)
            st.subheader("The Response is")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
