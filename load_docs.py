# load_docs.py 
#Adding Vector storage to not redo embedding again and again. Later will add a way to check if the vector embeddings exist for the same file
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Configuration ---
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
# Main directory to store all vector stores
VECTOR_STORES_DIR = "vector_stores"

def main():
    if not DRIVE_FOLDER_ID:
        print("❌ Error: DRIVE_FOLDER_ID not found in .env file.")
        return

    # specific path for this folder's index
    index_path = os.path.join(VECTOR_STORES_DIR, DRIVE_FOLDER_ID)

    # Check if the vector store already exists
    if os.path.exists(index_path):
        print(f"Vector store for folder '{DRIVE_FOLDER_ID}' already exists at '{index_path}'. No action needed.")
        return

    print(f"No existing store found. Starting the document processing pipeline for folder '{DRIVE_FOLDER_ID}'...")
    
    print(f"Loading documents from Google Drive folder...")
    loader = GoogleDriveLoader(folder_id=DRIVE_FOLDER_ID, credentials_path="credentials.json", token_path="token.json")
    docs = loader.load()

    if not docs:
        print("\n❌ CRITICAL: No documents were loaded from Google Drive.")
        return
    print(f"Loaded {len(docs)} document(s).")

    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        print("\n❌ CRITICAL: Documents were loaded, but failed to be split into chunks.")
        return
    print(f"Split documents into {len(chunks)} chunks.")

    print("\nCreating vector embeddings and saving to FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index to its specific directory 
    vectorstore.save_local(index_path)
    print(f"Vector store created and saved at: {index_path}")

    print("\nDocument processing pipeline completed successfully!")

if __name__ == "__main__":
    main()