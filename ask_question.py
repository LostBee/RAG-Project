# ask_question.py (FAISS version)
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # <-- Use FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"

def main():
    load_dotenv()

    print("Initializing models and loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    
    # Load the FAISS index
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("✅ Initialization complete. You can now ask questions.")

    # --- Interactive Loop ---
    while True:
        query = input("\nType your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
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