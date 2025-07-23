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

# Global variable to store document content
doc_content = ""
current_filename = ""

def upload_file(file):
    global doc_content, current_filename
    if file is None:
        return "No file uploaded", "Please upload a PDF or DOCX file to start chatting."
    
    try:
        file_data = extract_file_to_json(file.name)
        doc_content = file_data['content']
        current_filename = file_data['filename']
        return f"✅ Document '{current_filename}' loaded successfully!", "Document loaded. You can now ask questions about its content."
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", "Please try uploading a different file."

def chat_with_document(message, history):
    global doc_content, current_filename
    
    if not doc_content:
        return history + [[message, "⚠️ Please upload a document first before asking questions."]]
    
    if not message.strip():
        return history + [[message, "Please enter a question."]]
    
    try:
        start_time = time.time()
        answer = ask_query_on_chunks(doc_content, message)
        end_time = time.time()
        
        # Print resource usage to console (original functionality)
        print_resource_usage()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        # Format response with metadata
        response_time = f"\n\n*Response time: {end_time - start_time:.2f}s*"
        full_response = answer + response_time
        
        return history + [[message, full_response]]
    
    except Exception as e:
        error_msg = f"❌ Error processing your question: {str(e)}"
        return history + [[message, error_msg]]

def clear_chat():
    return []

def get_document_info():
    global current_filename, doc_content
    if not doc_content:
        return "No document loaded"
    
    word_count = len(doc_content.split())
    char_count = len(doc_content)
    return f"📄 **{current_filename}**\n\n📊 **Document Stats:**\n- Characters: {char_count:,}\n- Words: {word_count:,}\n- Estimated reading time: {word_count // 200} minutes"

# Create Gradio interface
with gr.Blocks(title="Document Chatbot", theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center; color: #2563eb;'>📚 Document Chatbot</h1>")
    gr.HTML("<p style='text-align: center; color: #64748b;'>Upload a PDF or DOCX file and ask questions about its content</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            gr.HTML("<h3>📎 Upload Document</h3>")
            file_input = gr.File(
                label="Select PDF or DOCX file",
                file_types=[".pdf", ".docx"],
                file_count="single"
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                lines=1,
                placeholder="No file uploaded yet..."
            )
            
            # Document info section
            gr.HTML("<h3>📋 Document Info</h3>")
            doc_info = gr.Markdown("No document loaded")
            
            # Update info button
            refresh_info_btn = gr.Button("🔄 Refresh Info", variant="secondary", size="sm")
        
        with gr.Column(scale=2):
            # Chat section
            gr.HTML("<h3>💬 Chat with Document</h3>")
            chatbot = gr.Chatbot(
                height=400,
                placeholder="Upload a document and start asking questions...",
                show_label=False,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
    
    # Example questions
    gr.HTML("<h3>💡 Example Questions</h3>")
    with gr.Row():
        example_btns = [
            gr.Button("What is the main topic?", variant="secondary", size="sm"),
            gr.Button("Summarize the key points", variant="secondary", size="sm"),
            gr.Button("What are the conclusions?", variant="secondary", size="sm"),
        ]
    
    # Event handlers
    file_input.upload(
        fn=upload_file,
        inputs=[file_input],
        outputs=[upload_status, doc_info]
    )
    
    submit_btn.click(
        fn=chat_with_document,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    msg_input.submit(
        fn=chat_with_document,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    refresh_info_btn.click(
        fn=get_document_info,
        outputs=[doc_info]
    )
    
    # Example button handlers
    for i, btn in enumerate(example_btns):
        example_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points discussed in this document?",
            "What are the main conclusions or findings mentioned in this document?"
        ]
        btn.click(
            lambda q=example_questions[i]: q,
            outputs=[msg_input]
        )

def main():
    # Original CLI functionality preserved
    print("Starting Document Chatbot...")
    print("Choose your interface:")
    print("1. Web Interface (Gradio) - Default")
    print("2. Command Line Interface")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        # Original CLI functionality
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
    else:
        # Launch Gradio interface
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )

if __name__ == "__main__":
    main()
1