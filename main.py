from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from crewai import Agent, Crew, Process
from crewai.tools import BaseTool
from typing import Any, Dict

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

#PDF Processing Class
class PDFReaderAgent:
    def __init__(self):
        self.vector_store = None

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text() or ""
                    text += extracted_text
            except Exception as e:
                return f"Error reading {pdf.name}: {e}"
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def create_vector_store(self, text_chunks):
        if not text_chunks:
            return "No text chunks available to create FAISS index."

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        self.vector_store.save_local("faiss_index")
        return "Vector store created successfully."

    def process_pdfs(self, pdf_docs):
        raw_text = self.get_pdf_text(pdf_docs)
        if raw_text:
            return self.create_vector_store(self.get_text_chunks(raw_text))
        return "No text was extracted from the PDFs."

    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the provided context, just say, "Answer is not available in the context." 
        Do not provide a wrong answer.\n\n
        Context:\n {context}\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def query_pdf(self, question):
        if os.path.exists("faiss_index/index.faiss"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(question)
            chain = self.get_conversational_chain()
            response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
            return response["output_text"]
        return "Please process PDF documents first."

# Image Processing Class
class ImageAnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def process_image(self, image_file, input_text, prompt=""):
        if image_file:
            bytes_data = image_file.getvalue()
            image_parts = [{"mime_type": image_file.type, "data": bytes_data}]
            try:
                response = self.model.generate_content([input_text, image_parts[0], prompt])
                return response.text
            except Exception as e:
                return f"Error analyzing image: {str(e)}"
        return "No image provided."
    
# Chatbot Class
class ChatbotAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.chat = self.model.start_chat(history=[])

    def respond_to_query(self, question):
        try:
            response = self.chat.send_message(question)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# CrewAI Tools
class ProcessPDFTool(BaseTool):
    name: str = "Process PDF Documents"
    description: str = "Extract text and create a searchable vector store from PDFs"

    def _run(self, pdf_docs: Any) -> str:
        return PDFReaderAgent().process_pdfs(pdf_docs)

class QueryPDFTool(BaseTool):
    name: str = "Query PDF Content"
    description: str = "Ask questions about processed PDF documents"

    def _run(self, question: str) -> str:
        return PDFReaderAgent().query_pdf(question)

class ProcessImageTool(BaseTool):
    name: str = "Analyze Image"
    description: str = "Analyze images and extract information"

    def _run(self, args: Dict[str, Any]) -> str:
        return ImageAnalysisAgent().process_image(
            args.get("image_file"), args.get("input_text", ""), args.get("prompt", "")
        )

class RespondToQueryTool(BaseTool):
    name: str = "Generate Text Response"
    description: str = "Provide a text response to user queries"

    def _run(self, question: str) -> str:
        return ChatbotAgent().respond_to_query(question)

# CrewAI Agents
def create_crew_ai_agents():
    return (
        Agent(
            role="PDF Reader Expert",
            goal="Extract and retrieve information from PDFs",
            backstory="An AI specialized in processing PDF documents.",
            verbose=True,
            allow_delegation=True,
            tools=[ProcessPDFTool(), QueryPDFTool()],
        ),
        Agent(
            role="Image Analyst",
            goal="Analyze and extract meaningful insights from images",
            backstory="An AI that can analyze images for relevant information.",
            verbose=True,
            allow_delegation=True,
            tools=[ProcessImageTool()],
        ),
        Agent(
            role="Conversational Agent",
            goal="Engage in natural conversations and provide insightful responses",
            backstory="A chatbot designed to answer user queries and provide assistance.",
            verbose=True,
            allow_delegation=True,
            tools=[RespondToQueryTool()],
        ),
    )

def create_crew(pdf_agent, image_agent, chatbot_agent):
    return Crew(
        agents=[pdf_agent, image_agent, chatbot_agent],
        tasks=[],
        verbose=True,
        process=Process.sequential,
    )

# Streamlit Application
def streamlit_app():
    st.set_page_config(page_title="AI Assistant Hub", layout="wide")

    # Initialize agents
    pdf_reader = PDFReaderAgent()
    image_analyzer = ImageAnalysisAgent()
    chatbot = ChatbotAgent()

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Chat with PDF", "Image Analysis", "ChatBot"])

    # Chat with PDF tab
    with tab1:
        st.header("Chat with PDF 📄")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Upload Documents")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

            if pdf_docs and st.button("Process PDFs"):
                with st.spinner("Processing your PDFs..."):
                    result = pdf_reader.process_pdfs(pdf_docs)
                    st.success(result)

        with col2:
            st.subheader("Ask Questions")
            pdf_question = st.text_input("Ask a question about your documents", key="pdf_question")

            if pdf_question:
                with st.spinner("Searching documents..."):
                    pdf_response = pdf_reader.query_pdf(pdf_question)
                    st.write("Response:")
                    st.write(pdf_response)

    # Image Analysis tab
    with tab2:
        st.header("Image Analysis 🖼️")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_file is not None:
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Uploaded Image", use_column_width=True)

            input_text = st.text_input("What would you like to know about this image?", key="image_query")

            input_prompt = """
            You are an expert in understanding images and documents.
            You will receive input images and you will have to answer questions based on the input image.
            """

            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    image_response = image_analyzer.process_image(uploaded_file, input_prompt, input_text)
                    st.subheader("Analysis Results")
                    st.write(image_response)

    # General Chat tab
    with tab3:
        st.header("Chat Assisant 💬")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for role, text in st.session_state["chat_history"]:
            st.write(f"**{role}:** {text}")

        chat_input = st.text_input("Type your message:")
        if st.button("Send") and chat_input:
            st.session_state["chat_history"].append(("You", chat_input))
            with st.spinner("Thinking..."):
                response = chatbot.respond_to_query(chat_input)
                st.session_state["chat_history"].append(("Assistant", response))
            st.rerun()

if __name__ == "__main__":
    pdf_agent, image_agent, chatbot_agent = create_crew_ai_agents()
    crew = create_crew(pdf_agent, image_agent, chatbot_agent)
    streamlit_app()
