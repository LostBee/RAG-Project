# ask_question.py (Main Application)
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from utils import get_id_from_path, process_documents # <-- Import from utils

load_dotenv()
VECTOR_STORES_DIR = "vector_stores"

def main():
    vectorstore = None
    print("\n--- RAG Document Chat ---")
    
    # NEW: Main user workflow
    use_existing = input("Have you previously processed the files you want to chat with? (yes/no): ").lower()
    
    print("\nSelect a data source:")
    print("1. Google Drive Folder")
    print("2. Local Folder")
    source_choice = input("Enter your choice (1 or 2): ")

    source_path = ""
    source_id = ""
    loader = None

    if source_choice == '1':
        source_path = input("Enter the Google Drive Folder ID: ")
        source_id = source_path
        loader = GoogleDriveLoader(folder_id=source_id, credentials_path="credentials.json", token_path="token.json")
    elif source_choice == '2':
        source_path = input("Enter the path to the local folder: ")
        source_id = get_id_from_path(source_path)
        loader = DirectoryLoader(source_path, glob="**/*.*", show_progress=True)
    else:
        print("Invalid choice. Exiting.")
        return

    index_path = os.path.join(VECTOR_STORES_DIR, source_id)
    
    if use_existing == 'yes':
        if os.path.exists(index_path):
            print("Loading existing knowledge base...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print("Knowledge base not found for this source.")
            create_new = input("Would you like to process it now? (yes/no): ").lower()
            if create_new == 'yes':
                vectorstore = process_documents(source_id, loader)
    elif use_existing == 'no':
        print("Forcing a refresh of the knowledge base.")
        vectorstore = process_documents(source_id, loader, force_refresh=True)

    if not vectorstore:
        print("Could not load or create a knowledge base. Exiting.")
        return

    # --- Chat Loop ---
    print("✅ Initialization complete. You can now ask questions.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    while True:
        query = input("\nType your question (or 'quit' to exit): ")
        if query.lower() == 'quit': break
        if not query.strip(): continue
        
        print("Thinking with Gemini Flash...")
        response = qa_chain.invoke(query)
        print("\n✅ Answer:")
        print(response["result"])

if __name__ == "__main__":
    main()