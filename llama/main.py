get_document_input
# main.py
from Extraction import get_document_input
from ui import launch_ui

if __name__ == "__main__":
    print("Indexing documents...")
    total = index_documents()
    print(f"{total} chunks indexed.")

    app = launch_ui()
    app.launch()
