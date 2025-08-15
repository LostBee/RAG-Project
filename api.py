# api.py (with .env loading)
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# --- NEW: Import load_dotenv ---
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from utils import get_id_from_path, create_vector_store

# --- NEW: Load environment variables from .env file ---
load_dotenv()

app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    source_type: str
    path: str
    question: str

VECTOR_STORES_DIR = "vector_stores"

@app.post("/ask")
def ask_question(request: QueryRequest):
    source_id = ""
    loader = None
    vectorstore = None
    
    if request.source_type == "drive":
        source_id = request.path
        loader = GoogleDriveLoader(folder_id=source_id, credentials_path="credentials.json", token_path="token.json")
    elif request.source_type == "local":
        if not os.path.isdir(request.path):
            return {"error": f"Invalid local path: {request.path} is not a directory."}
        source_id = get_id_from_path(request.path)
        loader = DirectoryLoader(request.path, glob="**/*.*", show_progress=True)
    else:
        return {"error": "Invalid source_type. Must be 'drive' or 'local'."}

    index_path = os.path.join(VECTOR_STORES_DIR, source_id)
    
    if os.path.exists(index_path):
        print(f"✅ Loading existing vector store from '{index_path}'")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"⚠️ No vector store found for source '{source_id}'. Creating a new one.")
        vectorstore = create_vector_store(source_id, loader)

    if not vectorstore:
        return {"error": "Could not create or load the knowledge base."}
    
    # Corrected the model name typo here
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    try:
        response = qa_chain.invoke(request.question)
        return {"answer": response["result"]}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}