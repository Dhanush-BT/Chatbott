import os
import fitz  
from docx import Document
import json
import ollama
from typing import Dict, Any
import time
import psutil
import gradio as gr

def print_resource_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    print(f" Memory: {mem_mb:.2f} MB |  CPU: {cpu_percent:.2f}%")

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_file_to_json(file_path: str) -> Dict[str, Any]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

    return {
        "filename": os.path.basename(file_path),
        "filepath": file_path,
        "content": text
    }

def chunk_text(text: str, max_chunk_size: int = 3500, overlap_size: int = 200):
    """
    Splits text into overlapping chunks for LLM processing.
    :param text: Full document text
    :param max_chunk_size: Maximum chunk size (characters)
    :param overlap_size: Number of overlapped characters between chunks
    :return: List of chunk strings
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap_size
        if start < 0:
            start = 0
    return chunks

def ask_query_on_chunks(doc_text: str, question: str):
    chunks = chunk_text(doc_text)
    answers = []
    for chunk in chunks:
        system_prompt = (
            "You are a helpful assistant. Only answer questions strictly based on the document provided below. "
            "If the answer is not in the document, say 'The document does not contain that information."
            "You can support greeting or casual conversation inputs, but always prioritize the document content.\n\n"
            f"DOCUMENT:\n{chunk}\n\n"
            f"QUESTION: {question}"
        )
        response = ollama.chat(
            model='gemma2:2b',
            messages=[{"role": "user", "content": system_prompt}]
        )
        answer = response['message']['content']
        answers.append(answer.strip())
        if answer.strip() and "does not contain" not in answer:
            # Return as soon as a relevant answer is found
            return answer.strip()
    # If no relevant answer, return combined or default message
    return "The answer was not found in the document."

def main():
    file_path = input("Enter the path to your PDF or DOCX file: ").strip()
    file_data = extract_file_to_json(file_path)
    doc_text = file_data['content']

    print("Document loaded. You can now ask questions based on its content.")
    print("Type 'exit' to quit.\n")

    while True:
    
        query = input("\n Ask a question: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        start_time = time.time()
        answer = ask_query_on_chunks(doc_text, query)
        print(f"\nAnswer: {answer}\n")
        print_resource_usage()
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
