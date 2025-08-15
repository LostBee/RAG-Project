# load_docs.py (with improved path validation)
import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTOR_STORES_DIR = "vector_stores"

def get_id_from_path(path):
    """Creates a short, safe, unique ID from a file path."""
    return hashlib.md5(path.encode()).hexdigest()

def process_documents(source_id, loader):
    """The main processing pipeline for any data source."""
    index_path = os.path.join(VECTOR_STORES_DIR, source_id)
    
    if os.path.exists(index_path):
        print(f"✅ Vector store for source '{source_id}' already exists. No action needed.")
        return

    print(f"🚀 No existing store found. Starting the processing pipeline for source '{source_id}'...")
    docs = loader.load()

    if not docs:
        print("\n❌ CRITICAL: No documents were loaded.")
        return
    print(f"✅ Loaded {len(docs)} document(s).")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if not chunks:
        print("\n❌ CRITICAL: Documents failed to be split into chunks.")
        return
    print(f"✅ Split documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vector store created and saved at: {index_path}")
    print("\n🎉 Document processing pipeline completed successfully!")

def main():
    while True:
        print("\nSelect a data source to process:")
        print("1. Google Drive Folder")
        print("2. Local Folder")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            folder_id = input("Enter your Google Drive Folder ID: ")
            if folder_id:
                loader = GoogleDriveLoader(folder_id=folder_id, credentials_path="credentials.json", token_path="token.json")
                process_documents(folder_id, loader)
            break
        elif choice == '2':
            local_path = input("Enter the path to your local folder: ")
            # --- IMPROVED VALIDATION ---
            if os.path.isdir(local_path): # Check if it's a directory
                loader = DirectoryLoader(local_path, glob="**/*.*", show_progress=True)
                source_id = get_id_from_path(local_path)
                process_documents(source_id, loader)
            elif os.path.isfile(local_path): # Check if it's a file
                 print("❌ Error: You provided a path to a file. Please provide a path to a folder.")
            else:
                print("❌ Error: The specified path does not exist.")
            break
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()