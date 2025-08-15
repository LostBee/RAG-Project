# api.py (Refactored for logical consistency)
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from utils import get_id_from_path, get_id_from_files, create_vector_store

load_dotenv()
app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

VECTOR_STORES_DIR = "vector_stores"

@app.post("/ask")
def ask_question(
    source_type: str = Form(...),
    question: str = Form(...),
    path: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    vectorstore = None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- REFACTORED LOGIC: Each source type is handled independently ---
    if source_type == "drive":
        if not path: return {"error": "Google Drive Folder ID is required."}
        source_id = path
        index_path = os.path.join(VECTOR_STORES_DIR, source_id)
        if os.path.exists(index_path):
            print(f"✅ Loading existing vector store from '{index_path}'")
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = GoogleDriveLoader(folder_id=source_id, credentials_path="credentials.json", token_path="token.json")
            vectorstore = create_vector_store(source_id, loader)

    elif source_type == "local":
        if not path: return {"error": "Local folder path is required."}
        if not os.path.isdir(path): return {"error": f"Invalid path: {path} is not a directory."}
        source_id = get_id_from_path(path)
        index_path = os.path.join(VECTOR_STORES_DIR, source_id)
        if os.path.exists(index_path):
            print(f"✅ Loading existing vector store from '{index_path}'")
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = DirectoryLoader(path, glob="**/*.*", show_progress=True)
            vectorstore = create_vector_store(source_id, loader)

    elif source_type == "upload":
        if not files: return {"error": "No files were uploaded."}
        source_id = get_id_from_files(files)
        index_path = os.path.join(VECTOR_STORES_DIR, source_id)
        if os.path.exists(index_path):
            print(f"✅ Loading existing vector store from '{index_path}'")
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in files:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        f.write(file.file.read())
                loader = DirectoryLoader(temp_dir, glob="**/*.*", show_progress=True)
                vectorstore = create_vector_store(source_id, loader)
    else:
        return {"error": "Invalid source_type specified."}

    if not vectorstore: 
        return {"error": "Could not create or load the knowledge base."}
    
    # --- QA Chain Logic (runs after a vectorstore is successfully loaded/created) ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    try:
        response = qa_chain.invoke(question)
        return {"answer": response["result"]}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}