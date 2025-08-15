# utils.py
import os
import hashlib
import shutil
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORES_DIR = "vector_stores"

def get_id_from_path(path_or_id):
    """Creates a short, safe, unique ID from a file path."""
    source_id = hashlib.md5(path_or_id.encode()).hexdigest()
    print(f"--- DEBUG: Path='{path_or_id}'  -->  Hashed ID='{source_id}' ---")
    return source_id

def create_vector_store(source_id, loader):
    """
    Creates a new vector store from a loader, saves it, and returns it.
    """
    index_path = os.path.join(VECTOR_STORES_DIR, source_id)

    # Clean up old directory if it exists for a fresh start
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    
    print(f"🚀 Creating new vector store for source '{source_id}'...")
    docs = loader.load()

    if not docs:
        print("\n❌ CRITICAL: No documents were loaded.")
        return None
    print(f"✅ Loaded {len(docs)} document(s).")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if not chunks:
        print("\n❌ CRITICAL: Documents failed to be split into chunks.")
        return None
    print(f"✅ Split documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vector store created and saved at: {index_path}")
    
    return vectorstore