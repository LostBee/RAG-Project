# check_db.py (FAISS version)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index"

def check_database():
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Load the FAISS index
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            # This is needed to allow loading from a local file
            allow_dangerous_deserialization=True
        )
        
        # Get the number of documents in the FAISS index
        doc_count = len(vectorstore.index_to_docstore_id)
        
        print(f"\n✅ Success! The FAISS index contains {doc_count} document chunks.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   This might mean the index directory doesn't exist.")
        print("   Make sure you have run 'python load_docs.py' successfully.")

if __name__ == "__main__":
    check_database()