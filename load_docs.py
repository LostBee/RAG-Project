# load_docs.py

from langchain_community.document_loaders import GoogleDriveLoader
import os

# --- Configuration ---
DRIVE_FOLDER_ID = "1sSoTFq02iXSiqm_NGUoCYSOVqFx2NIJQ" 

# --- Main Execution ---
def main():
    """
    Connects to Google Drive, loads documents from the specified folder,
    and prints the number of documents loaded and the content of the first one.
    """
    #
    # THIS IS THE LINE TO FIX. Change it back to check for the placeholder.
    #
    if DRIVE_FOLDER_ID == "YOUR_FOLDER_ID_HERE" or not DRIVE_FOLDER_ID:
        print("ERROR: Please set the DRIVE_FOLDER_ID variable in the script.")
        return

    print(f"Attempting to load documents from Google Drive folder: {DRIVE_FOLDER_ID}")

    # Initialize the GoogleDriveLoader
    loader = GoogleDriveLoader(
        folder_id=DRIVE_FOLDER_ID,
        credentials_path="credentials.json", #Pointing explicitly to the file to fix issues.
        token_path="token.json" # Have to specify as it tried to save it in a folder in C drive which doesn't exit.
    )

    # Load documents
    try:
        docs = loader.load()
        if not docs:
            print("No documents found in the specified folder. Make sure the folder is not empty and the ID is correct.")
            return

        print(f"\n✅ Successfully loaded {len(docs)} document(s).")
        print("\n--- Content of the first document: ---")
        print(docs[0].page_content[:500])
        print("\n--- Metadata of the first document: ---")
        print(docs[0].metadata)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the following:")
        print("1. Is your DRIVE_FOLDER_ID correct?")
        print("2. Is the 'credentials.json' file in the same directory as this script?")
        print("3. Did you enable the Google Drive API in your Google Cloud project?")


if __name__ == "__main__":
    main()