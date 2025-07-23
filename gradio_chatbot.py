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
    total_chunks = len(chunks)
    matched_chunk_index = None
    chunk_used_length = 0

    for i, chunk in enumerate(chunks):
        system_prompt = (
            "You are a helpful assistant. Only answer questions strictly based on the document provided below. "
            "If the answer is not in the document, say 'The document does not contain that information.' "
            "You can support greeting or casual conversation inputs, but always prioritize the document content.\n\n"
            f"DOCUMENT:\n{chunk}\n\n"
            f"QUESTION: {question}"
        )
        response = ollama.chat(
            model='gemma2:2b',
            messages=[{"role": "user", "content": system_prompt}]
        )
        answer = response['message']['content'].strip()
        answers.append(answer)

        if answer and "does not contain" not in answer:
            matched_chunk_index = i
            chunk_used_length = len(chunk)
            return answer, {
                "matched_chunk": i + 1,
                "total_chunks": total_chunks,
                "chunk_length": chunk_used_length,
                "source": " Matched"
            }

    # If no good match found
    return "The answer was not found in the document.", {
        "matched_chunk": None,
        "total_chunks": total_chunks,
        "chunk_length": 0,
        "source": " Fallback (No match found)"
    }


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
        return f"Document '{current_filename}' loaded successfully!", f" **{current_filename}**\n\n **Document loaded and ready for questions!**"
    except Exception as e:
        return f"Error loading file: {str(e)}", "Please try uploading a different file."

def chat_with_document(message, history):
    global doc_content

    if not doc_content:
        return history + [[message, " Please upload a PDF or DOCX document using the file upload area above before asking questions."]]
    
    if not message.strip():
        return history + [[message, "Please enter a question."]]

    try:
        start_time = time.time()
        answer, metadata = ask_query_on_chunks(doc_content, message)
        end_time = time.time()

        # Resource usage
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 ** 2)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        response_time = end_time - start_time

        # Metadata markdown block
        meta_info = (
            f"\n---\n"
            f" **Answer Source:** {metadata['source']}\n"
            f" **Chunk Used:** {metadata['matched_chunk']}/{metadata['total_chunks'] if metadata['total_chunks'] else 'N/A'}\n"
            f" **Chunk Length:** {metadata['chunk_length']} characters\n"
            f" **CPU Usage:** {cpu_percent:.2f}% &nbsp;&nbsp;&nbsp; **Memory:** {mem_mb:.2f} MB\n"
            f" **Response Time:** {response_time:.2f} seconds"
        )

        return history + [[message, answer + meta_info]]

    except Exception as e:
        return history + [[message, f" Error processing your question: {str(e)}"]]


def clear_chat():
    return []

def get_document_info():
    global current_filename, doc_content
    if not doc_content:
        return "No document loaded"
    
    word_count = len(doc_content.split())
    char_count = len(doc_content)
    return f" **{current_filename}**\n\n **Document Stats:**\n- Characters: {char_count:,}\n- Words: {word_count:,}\n- Estimated reading time: {word_count // 200} minutes"

# Create Gradio interface
with gr.Blocks(title="DocuMind", theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center; color: #2563eb;'> DocuMind </h1>")
    gr.HTML("<p style='text-align: center; color: #64748b;'>Upload a PDF or DOCX file and ask questions about its content</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            gr.HTML("<h3> Upload Document</h3>")
            file_input = gr.File(
                label="Select PDF or DOCX file",
                file_types=[".pdf", ".docx"],
                file_count="single"
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                lines=1,
                value="No file uploaded yet..."
            )
            
            # Document info section
            gr.HTML("<h3> Document Info</h3>")
            doc_info = gr.Markdown("No document loaded")
        
        with gr.Column(scale=2):
            # Chat section
            gr.HTML("<h3> Chat with Document</h3>")
            chatbot = gr.Chatbot(
                height=400,
                show_label=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button(" Clear Chat", variant="secondary", size="sm")
    
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

if __name__ == "__main__":
    demo.launch(
        share=True,
        inbrowser=True
    )
    