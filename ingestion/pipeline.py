import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def process_and_index_pdf(pdf_path: str, ticker: str, doc_type: str = "10-K", persist_dir: str = "./chroma_db"):
    """
    Extracts pages from a PDF, splits them into semantic chunks, enriches them with metadata,
    and indexes them into ChromaDB using local HuggingFace embeddings.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"Loading PDF pages from {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Splitting {len(pages)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    # Enrich metadata for tracking and filtering
    for c in chunks:
        c.metadata.update({
            "ticker": ticker.upper(),
            "doc_type": doc_type,
            "source_file": os.path.basename(pdf_path)
        })

    print(f"Initializing embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print(f"Persisting {len(chunks)} chunks into ChromaDB at {persist_dir}...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print("Ingestion and indexing completed successfully!")
    return vectordb

if __name__ == "__main__":
    # Test execution using a sample path if present
    sample_pdf = "data/AAPL_10K_2025.pdf"
    if os.path.exists(sample_pdf):
        process_and_index_pdf(sample_pdf, ticker="AAPL", doc_type="10-K")
    else:
        print(f"Place a sample PDF at '{sample_pdf}' to run ingestion locally.")