# ask_question.py
# Checkout this awesome playlist: https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"

def main():
    # Load API keys from .env file
    load_dotenv()

    # Initialize Models
    print("Initializing models...")
    # Initialize the same embedding model used to create the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Initialize Gemini 2.5 Flash for generation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print("✅ Models initialized.")

    # Load the Vector Store
    print(f"Loading vector store from: {FAISS_INDEX_PATH}")
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("   Please make sure you have run 'python load_docs.py' to create the index.")
        return
    print("✅ Vector store loaded.")

    # Create the QA Chain
    # A retriever is a component that fetches relevant documents from the vector store - checkout the youtube playlist on top
    retriever = vectorstore.as_retriever()
    
    # The RetrievalQA chain combines the retriever and the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means it "stuffs" all retrieved docs into the prompt
        retriever=retriever
    )
    print("✅ QA chain created.")

    # Ask a Question
    print("\nReady to answer questions.")
    query = "What is 2+2 based on the document?" # <-- CHANGE YOUR QUESTION HERE
    
    print(f"\n❓ Query: {query}")
    try:
        response = qa_chain.invoke(query)
        print("\n✅ Answer:")
        print(response["result"])
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()