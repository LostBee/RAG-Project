# ask_question.py (using user-specified model)
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

# --- Configuration ---
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
VECTOR_STORES_DIR = "vector_stores"

def main():
    if not DRIVE_FOLDER_ID:
        print("❌ Error: DRIVE_FOLDER_ID not found in .env file. Cannot select a database.")
        return

    index_path = os.path.join(VECTOR_STORES_DIR, DRIVE_FOLDER_ID)
    if not os.path.exists(index_path):
        print(f"❌ Error: Vector store for folder '{DRIVE_FOLDER_ID}' does not exist.")
        print(f"   Please run 'python load_docs.py' first with this ID in your .env file.")
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

        try:
            print("Thinking with Gemini...")
            response = qa_chain.invoke(query)
            print("\n✅ Answer:")
            print(response["result"])
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()