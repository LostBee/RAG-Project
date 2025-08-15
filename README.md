# RAG Document Chatbot

This project is a Python-based implementation of a Retrieval-Augmented Generation (RAG) system. The goal is to create a chatbot that can answer questions based on the content of various document sources.

The application is built with a separate backend API (using FastAPI and LangChain) and a frontend web interface (using HTML, CSS, and JavaScript), making it modular and extensible.

---

## Project Status

**Completed:** The project is a fully functional web application with a local backend API.

### Features

* **Multiple Data Sources:** Ingest documents from a Google Drive folder, a local folder on your machine, or by direct file upload.
* **Web Interface:** A clean, browser-based UI for selecting a data source, processing documents, and asking questions.
* **Dynamic UI:** The interface dynamically adjusts to show the relevant inputs for the selected data source.
* **Backend API:** Built with FastAPI, providing a robust endpoint that handles all RAG logic, including file uploads.
* **Persistent Vector Storage:** Automatically creates and manages local vector stores for each data source using FAISS, allowing for quick re-use of processed knowledge bases.
* **Local Embeddings:** Uses a free, high-performance Hugging Face model (`all-MiniLM-L6-v2`) to create vector embeddings locally.
* **LLM Generation:** Leverages Google's Gemini Flash Large Language Model to generate context-aware answers.
* **Secure Configuration:** Manages all API keys using a `.env` file, keeping secrets out of the code.
* **File Upload Validation:** Limits direct file uploads to a maximum of 5 files for a better user experience.

---

## Project Structure

* `api.py`: The FastAPI backend server that contains all the API logic.
* `index.html`: A single file containing the complete frontend UI, styling (CSS), and client-side logic (JavaScript).
* `utils.py`: Shared helper functions used by the backend for processing documents and creating unique IDs.
* `vector_stores/`: A directory that is automatically created to store the generated knowledge bases. This folder should be in your `.gitignore`.
* `.env`: A file to store your secret API keys.

---

## Usage

The application consists of a backend server and a frontend interface.

### Step 1: Start the Backend Server

Navigate to the project directory in your terminal (with the virtual environment activated) and run the following command to start the FastAPI server:

```bash
uvicorn api:app --reload
````

Keep this terminal window open. You will see log output here as you use the application.

### Step 2: Open the Web Interface

Simply find the `index.html` file in the project folder and open it in your web browser. Once loaded, you can choose your data source (Google Drive, Local Folder, or File Upload), provide the necessary path or files, and ask questions.

-----

## Setup and Installation

### 1\. Prerequisites

  * Python 3.8+ installed.
  * Access to a Google account.

### 2\. Clone the Repository

```bash
git clone <your-repo-url>
cd my-rag-project
```

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

*(Note: To create or update the `requirements.txt` file, run `pip freeze > requirements.txt` after installing all packages).*

### 5\. Configuration and Credentials

Create a file named `.env` in the root of the project folder to store your secrets.

**1. Google Drive API (`credentials.json`)** (Optional, if you use Google Drive)

  * Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
  * Enable the **Google Drive API** for your project.
  * Create an **OAuth 2.0 Client ID** credential for a **Desktop app**.
  * Download the credentials JSON file and save it as `credentials.json` in the project folder.

**2. Create `.env` File**
Create a file named `.env` and add the following keys:

```env
# For Google Gemini API
GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_KEY_HERE"

# For LangSmith Tracing (Optional, but recommended)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY_HERE"
LANGCHAIN_PROJECT="My RAG Project"
```

  * **`GOOGLE_API_KEY`**: Get this from [Google AI Studio](https://aistudio.google.com/app/apikey).
  * **`LANGCHAIN_API_KEY`**: Get this from the [LangSmith website](https://smith.langchain.com/).

<!-- end list -->
