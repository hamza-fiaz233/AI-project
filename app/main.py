import os
import sys
# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()  # Automatically loads OPENAI_API_KEY from the .env file

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from agent.graph import app as agent_app

# Page Configuration
st.set_page_config(
    page_title="FinAI Analyst",
    page_icon="📈",
    layout="wide"
)

# Custom CSS Styling for a professional financial UI look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1 {
        color: #58a6ff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 FinAI Analyst: Agentic RAG Financial Research Assistant")
st.caption("Combine real-time market data (yfinance) with SEC document intelligence & ChromaDB.")[cite: 7]

# Sidebar Configuration for Session, Ticker, and PDF Upload & Auto-Indexing
with st.sidebar:
    st.header("⚙️ Session Settings")
    thread_id = st.text_input("Thread ID / Session Name", value="default_investor")
    
    st.divider()
    st.subheader("📁 Upload SEC / Financial PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_pdf is not None:
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_pdf.name)
        
        # Save file to disk
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        # Automatically process and index the PDF into ChromaDB (Module A Pipeline)
        with st.spinner("Processing and indexing document into ChromaDB..."):
            try:
                loader = PyMuPDFLoader(pdf_path)
                pages = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)[cite: 12]
                chunks = splitter.split_documents(pages)
                
                for c in chunks:
                    c.metadata.update({
                        'ticker': 'CUSTOM', 
                        'doc_type': 'Uploaded_PDF', 
                        'source_file': uploaded_pdf.name
                    })
                
                embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')[cite: 12, 13]
                vectordb = Chroma.from_documents(
                    chunks, 
                    embeddings, 
                    persist_directory='./chroma_db'
                )
                st.success(f"Successfully indexed {uploaded_pdf.name} into vector store!")
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
    
    st.divider()
    st.subheader("📊 Target Asset Info")
    target_ticker = st.text_input("Primary Ticker", value="AAPL").upper()
    doc_type = st.selectbox("SEC Document Type", ["10-K", "10-Q", "Transcript"])[cite: 11]
    
    st.divider()
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.messages = []
        st.success("History cleared for UI session!")

# Initialize session state messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from Streamlit session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input(f"Ask a question about {target_ticker} or financial markets..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke LangGraph Agent with thread config
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Analyzing market data and searching SEC filings...")
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=prompt)]}
            response_state = agent_app.invoke(inputs, config=config)
            
            ai_messages = [m for m in response_state["messages"] if isinstance(m, AIMessage)]
            latest_response = ai_messages[-1].content if ai_messages else "No response generated."
            
            message_placeholder.markdown(latest_response)
            st.session_state.messages.append({"role": "assistant", "content": latest_response})
            
        except Exception as e:
            error_msg = f"An error occurred during agent execution: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})