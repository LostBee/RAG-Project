# load_docs.py (Force Refresh Tool)
import os
from langchain_community.document_loaders import GoogleDriveLoader, DirectoryLoader
from utils import get_id_from_path, process_documents # <-- Import from utils

def main():
    print("\n--- Force Refresh Knowledge Base ---")
    print("This will delete any existing data for the source and re-process it.")
    print("1. Google Drive Folder")
    print("2. Local Folder")
    choice = input("Select a data source to refresh (1 or 2): ")

    if choice == '1':
        folder_id = input("Enter your Google Drive Folder ID: ")
        if folder_id:
            loader = GoogleDriveLoader(folder_id=folder_id, credentials_path="credentials.json", token_path="token.json")
            process_documents(folder_id, loader, force_refresh=True)
    elif choice == '2':
        local_path = input("Enter the path to your local folder: ")
        if os.path.isdir(local_path):
            loader = DirectoryLoader(local_path, glob="**/*.*", show_progress=True)
            source_id = get_id_from_path(local_path)
            process_documents(source_id, loader, force_refresh=True)
        else:
            print("❌ Error: The specified path is not a valid directory.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()