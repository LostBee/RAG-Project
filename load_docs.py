# load_docs.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==DEBUGGING to solve encoding issue!!- Solved and I don't even know what was the issue but leaving the logs here in case needed ==
print("--- Starting Debug ---")
# 1. Print the current working directory
print(f"Current Working Directory: {os.getcwd()}")

# 2. Check if the .env file exists at the expected path
env_path = os.path.join(os.getcwd(), '.env')
print(f"Checking for .env file at: {env_path}")
print(f"Does .env file exist? {os.path.exists(env_path)}")

# 3. Load the .env file and check if it was successful
load_success = load_dotenv()
print(f"Did dotenv load successfully? {load_success}")

# 4. Read the variable AFTER trying to load it
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
print(f"Value of DRIVE_FOLDER_ID after loading: '{DRIVE_FOLDER_ID}'")
print("--- End Debug ---\n")
# =========================================================


# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"

def main():
    if not DRIVE_FOLDER_ID:
        print("❌ Error: DRIVE_FOLDER_ID is still None. Halting execution.")
        return

    print("🚀 Starting the document processing pipeline...")
    # ... rest of the script is the same ...
    print(f"Loading documents from Google Drive folder...")
    loader = GoogleDriveLoader(
        folder_id=DRIVE_FOLDER_ID,
        credentials_path="credentials.json",
        token_path="token.json"
    )
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document(s).")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Split documents into {len(chunks)} chunks.")

    print("Creating vector embeddings and saving to FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ Vector store created and saved at: {FAISS_INDEX_PATH}")

    print("\n🎉 Document processing pipeline completed successfully!")

if __name__ == "__main__":
    main()