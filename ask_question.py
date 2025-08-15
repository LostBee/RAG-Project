# ask_question.py (with data source selection)
import os
import hashlib
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

VECTOR_STORES_DIR = "vector_stores"

def get_id_from_path(path):
    """Creates a short, safe, unique ID from a file path."""
    return hashlib.md5(path.encode()).hexdigest()

def main():
    source_id = None
    while True:
        print("\nSelect a data source to chat with:")
        print("1. Google Drive Folder")
        print("2. Local Folder")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            source_id = input("Enter the Google Drive Folder ID of the knowledge base: ")
            break
        elif choice == '2':
            local_path = input("Enter the path to the local folder of the knowledge base: ")
            if os.path.exists(local_path):
                source_id = get_id_from_path(local_path)
            else:
                print("❌ Error: The specified local path does not exist.")
            break
        elif choice == '3':
            return
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    if not source_id:
        return

    index_path = os.path.join(VECTOR_STORES_DIR, source_id)
    if not os.path.exists(index_path):
        print(f"❌ Error: Vector store for source '{source_id}' does not exist.")
        print(f"   Please run 'python load_docs.py' first to process this source.")
        return

    print("Initializing models and loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("✅ Initialization complete. You can now ask questions.")

    while True:
        query = input("\nType your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            print("Exiting chat. Goodbye!")
            break
        if not query.strip():
            continue
        
        print("Thinking with Gemini Flash...")
        response = qa_chain.invoke(query)
        print("\n✅ Answer:")
        print(response["result"])

if __name__ == "__main__":
    main()