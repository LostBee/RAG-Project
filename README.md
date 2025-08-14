
# Google Drive RAG Chatbot

This project is a Python-based implementation of a Retrieval-Augmented Generation (RAG) system. The goal is to create a chatbot that can answer questions based on the content of documents stored in a specific Google Drive folder.

The application uses the LangChain framework to orchestrate the connection to data sources (Google Drive), data processing, and interaction with a Large Language Model (LLM).

---

## Project Status

The project is currently in **Phase 1**.

### Completed Work (Phase 1)

* **Environment Setup:** The project structure is established with a dedicated Python virtual environment to manage dependencies.
* **Google Drive API Integration:** Secure authentication with Google Drive is set up using OAuth 2.0. The system can connect to a user's Google account.
* **Data Ingestion:** A Python script (`load_docs.py`) successfully loads documents from a specified Google Drive folder using `langchain_google_community.document_loaders.GoogleDriveLoader`.

---

## WIP / Future Phases

* **Phase 2: Data Processing & Vectorization**
    * Split the loaded documents into smaller, semantically meaningful chunks.
    * Use an embedding model (e.g., from OpenAI) to convert these chunks into numerical vectors.

* **Phase 3: Retrieval & Generation**
    * Store the vectors in a vector database for efficient searching.
    * Build a QA (Question-Answering) chain that takes a user query, finds the most relevant document chunks from the vector store, and passes them to an LLM to generate a factual answer.

* **Phase 4: Building the User Interface**
    * Develop a simple Command-Line Interface (CLI) that allows a user to interactively ask questions and receive answers.

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites
* Python 3.8+ installed.
* Access to a Google account and Google Drive.

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd my-rag-project
````

### 3\. Set Up Virtual Environment (Windows)

```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate
```

### 4\. Install Dependencies

Install the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Note: To create the `requirements.txt` file, run `pip freeze > requirements.txt` after installing the packages from the previous step).*

### 5\. Google Drive API Credentials

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
2.  Enable the **Google Drive API** for your project.
3.  Create an **OAuth 2.0 Client ID** credential for a **Desktop app**.
4.  Download the credentials JSON file and save it as `credentials.json` in the root of this project folder.

### 6\. Configure the Project

1.  Create a folder in your Google Drive (e.g., `RAG_Docs`) and add the documents you want the chatbot to use.
2.  Copy the **Folder ID** from the Google Drive URL.
3.  Open the `load_docs.py` file and paste your Folder ID into the `DRIVE_FOLDER_ID` variable.

### 7\. Run the Application

To test the document loading from Phase 1, run the following command in your terminal:

```bash
python load_docs.py
```

The first time you run this, you will be prompted to authorize access to your Google Account via a browser window.


