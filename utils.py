# utils.py (with hashing debug)
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
    # --- NEW DEBUG LINE ---
    print(f"--- DEBUG: Path='{path_or_id}'  -->  Hashed ID='{source_id}' ---")
    return source_id

def process_documents(source_id, loader, force_refresh=False):
    """
    The main processing pipeline. Loads, chunks, embeds, and saves documents.
    Set force_refresh to True to delete and rebuild an existing store.
    """
    index_path = os.path.join(VECTOR_STORES_DIR, source_id)
    
    if os.path.exists(index_path):
        if force_refresh:
            print(f"Force refresh enabled. Deleting existing vector store at '{index_path}'...")
            shutil.rmtree(index_path)
        else:
            print(f"✅ Vector store for source '{source_id}' already exists. Loading existing store.")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print(f"🚀 Starting the processing pipeline for source '{source_id}'...")
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