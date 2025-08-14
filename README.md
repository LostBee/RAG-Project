# Google Drive RAG Chatbot

This project is a Python-based implementation of a Retrieval-Augmented Generation (RAG) system. The goal is to create a chatbot that can answer questions based on the content of documents stored in a specific Google Drive folder.

The application uses the LangChain framework to orchestrate the connection to data sources (Google Drive), data processing, and interaction with a Large Language Model (LLM).

---

## Project Status

**Completed for Terminal use case:** The application is fully functional and all core phases are complete.

### Features

* **Data Ingestion:** Securely loads documents from a specified Google Drive folder using the Google Drive API.
* **Document Processing:** Automatically splits documents into smaller, semantically meaningful chunks for efficient processing.
* **Embeddings:** Uses a free, high-performance Hugging Face model (`all-MiniLM-L6-v2`) to create vector embeddings locally, requiring no embedding API costs.
* **Vector Storage:** Stores document embeddings in a persistent, local FAISS vector store for fast and efficient similarity searches.
* **Generation:** Leverages Google's Gemini Pro Large Language Model to generate context-aware answers.
* **Interactive CLI:** Provides a simple, interactive Command-Line Interface (CLI) for a continuous question-and-answer session.
* **Secure Configuration:** Manages all API keys, tokens, and IDs securely using a `.env` file, keeping secrets out of the source code.
* **Debugging & Tracing:** Integrated with LangSmith for full visibility and debugging of the RAG pipeline.

**Future** A Local Web interface with possibility to easily change LLM api from the interface maybe if that doesn't add any big security risk. I will research

---

## How to Run the Application

There is a two-step process to run the application.

### Step 1: Process Your Documents (Run Once)

First, you need to run the `load_docs.py` script. This will connect to your Google Drive, process all the documents, and save them into the local FAISS vector store.

```bash
python load_docs.py
````

*You only need to run this script once, or whenever you add, remove, or change the documents in your Google Drive folder.*

### Step 2: Start the Chatbot

Once your documents are processed, you can start the interactive chatbot.

```bash
python ask_question.py
```

This will load the local vector store and the models, then prompt you to ask questions.

-----

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

  * Python 3.8+ installed.
  * Access to a Google account and Google Drive.

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

*(Note: To create the `requirements.txt` file, run `pip freeze > requirements.txt` after installing all packages).*

### 5\. Configuration and Credentials

This project requires several credentials. Create a file named `.env` in the root of the project folder to store them.

**1. Google Drive API (`credentials.json`)**

  * Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
  * Enable the **Google Drive API** for your project.
  * Create an **OAuth 2.0 Client ID** credential for a **Desktop app**.
  * Download the credentials JSON file and save it as `credentials.json` in the project folder.
  * When you run the app for the first time, you will be prompted to authorize access, which will create a `token.json` file. Make sure `credentials.json` and `token.json` are listed in your `.gitignore` file.

**2. Create `.env` File**
Create a file named `.env` and add the following keys:

```env
# For Google Drive and Gemini
DRIVE_FOLDER_ID="YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE"
GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_KEY_HERE"

# For LangSmith Tracing (Optional, but recommended)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY_HERE"
LANGCHAIN_PROJECT="My RAG Project"
```

  * **`DRIVE_FOLDER_ID`**: Open the folder in Google Drive and copy the ID from the URL.
  * **`GOOGLE_API_KEY`**: Get this from [Google AI Studio](https://aistudio.google.com/app/apikey).
  * **`LANGCHAIN_API_KEY`**: Get this from the [LangSmith website](https://smith.langchain.com/).

<!-- end list -->

