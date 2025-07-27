import os
import fitz
from docx import Document
import json
import ollama
from typing import Dict, Any, List
import time
import psutil
import gradio as gr
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models as qmodels
from uuid import uuid4
import numpy as np


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


# Initialize embedding model and Qdrant client

embedder = SentenceTransformer("all-MiniLM-L6-v2")  
qclient = QdrantClient(":memory:")  


collection_name = "documind_chunks"
if not qclient.collection_exists(collection_name):
    qclient.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
    )


def store_chunks_in_qdrant(chunks: List[str], filename: str):
    """
    Store document chunks as vectors in Qdrant
    """

    vectors = embedder.encode(chunks, show_progress_bar=False)

    payloads = [
        {
            "filename": filename,
            "chunk_id": idx,
            "text": chunk,
            "chunk_length": len(chunk)
        }
        for idx, chunk in enumerate(chunks)
    ]
    
    points = [
        qmodels.PointStruct(
            id=str(uuid4()),
            vector=vec.tolist(),
            payload=payload
        )
        for vec, payload in zip(vectors, payloads)
    ]
    

    qclient.upsert(collection_name=collection_name, points=points)
    
    return len(points)


def search_similar_chunks(question: str, limit: int = 3):
    """
    Search for similar chunks using vector similarity
    """
 
    question_vector = embedder.encode(question)
    

    hits = qclient.search(
        collection_name=collection_name,
        query_vector=question_vector.tolist(),
        limit=limit
    )
    
    return hits


def ask_query_with_qdrant(question: str):
    """
    Answer question using Qdrant vector search
    """

    hits = search_similar_chunks(question, limit=3)
    
    if not hits:
        return "No relevant content found in the document.", {
            "matched_chunks": 0,
            "total_chunks": 0,
            "chunk_length": 0,
            "source": "No Match",
            "similarity_scores": []
        }

    context_chunks = []
    similarity_scores = []
    total_length = 0
    
    for hit in hits:
        context_chunks.append(hit.payload["text"])
        similarity_scores.append(round(hit.score, 3))
        total_length += hit.payload["chunk_length"]
    
    context = "\n\n".join(context_chunks)
    

    system_prompt = (
        "You are a helpful assistant. Answer the question based strictly on the document context provided below. "
        "Provide exact content from the document when possible. If possible, provide tabulated answers with 100% accuracy. "
        "If the answer is not in the document context, say 'The document does not contain that information.'\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    

    response = ollama.chat(
        model='gemma2:2b',
        messages=[{"role": "user", "content": system_prompt}]
    )
    
    answer = response['message']['content'].strip()
    
    metadata = {
        "matched_chunks": len(hits),
        "total_chunks": qclient.count(collection_name=collection_name).count,
        "chunk_length": total_length,
        "source": "Vector Search",
        "similarity_scores": similarity_scores
    }
    
    return answer, metadata


def clear_qdrant_collection():
    """
    Clear all vectors from the collection
    """
    try:
        qclient.delete_collection(collection_name=collection_name)
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
        )
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False



doc_content = ""
current_filename = ""
chunks_stored = 0


def upload_file(file):
    global doc_content, current_filename, chunks_stored
    
    if file is None:
        return "No file uploaded", "Please upload a PDF or DOCX file to start chatting."

    try:

        clear_qdrant_collection()

        file_data = extract_file_to_json(file.name)
        doc_content = file_data['content']
        current_filename = file_data['filename']
        
        
        chunks = chunk_text(doc_content)
        
        
        chunks_stored = store_chunks_in_qdrant(chunks, current_filename)
        
        status_msg = f"Document '{current_filename}' loaded successfully! {chunks_stored} chunks vectorized."
        info_msg = f"**{current_filename}**\n\n**Document loaded and ready for questions!**\n\n**Chunks stored:** {chunks_stored}"
        
        return status_msg, info_msg
        
    except Exception as e:
        return f"Error loading file: {str(e)}", "Please try uploading a different file."


def chat_with_document(message, history):
    global doc_content

    if not doc_content:
        return history + [[message, "Please upload a PDF or DOCX document using the file upload area above before asking questions."]]

    if not message.strip():
        return history + [[message, "Please enter a question."]]

    try:
        start_time = time.time()
        
        
        answer, metadata = ask_query_with_qdrant(message)
        
        end_time = time.time()

        
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 ** 2)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        response_time = end_time - start_time

        
        meta_info = (
            f"\n---\n"
            f"**Answer Source:** {metadata['source']}\n"
            f"**Chunks Used:** {metadata['matched_chunks']}/{metadata['total_chunks']}\n"
            f"**Total Chunk Length:** {metadata['chunk_length']} characters\n"
            f"**Similarity Scores:** {metadata.get('similarity_scores', 'N/A')}\n"
            f"**CPU Usage:** {cpu_percent:.2f}% &nbsp;&nbsp;&nbsp; **Memory:** {mem_mb:.2f} MB\n"
            f"**Response Time:** {response_time:.2f} seconds"
        )

        return history + [[message, answer + meta_info]]

    except Exception as e:
        return history + [[message, f"Error processing your question: {str(e)}"]]


def clear_chat():
    return []


def get_document_info():
    global current_filename, doc_content, chunks_stored
    if not doc_content:
        return "No document loaded"

    word_count = len(doc_content.split())
    char_count = len(doc_content)
    
    
    collection_info = qclient.get_collection(collection_name=collection_name)
    vector_count = collection_info.points_count
    
    return f"**{current_filename}**\n\n**Document Stats:**\n- Characters: {char_count:,}\n- Words: {word_count:,}\n- Estimated reading time: {word_count // 200} minutes\n- Vector chunks stored: {vector_count}\n- Collection status: Active"



with gr.Blocks(title="DocuMind", theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center; color: #2563eb;'>DocuMind</h1>")
    gr.HTML("<p style='text-align: center; color: #64748b;'>Upload a PDF or DOCX file and ask questions using semantic vector search powered by Qdrant</p>")

    with gr.Row():
        with gr.Column(scale=1):
            
            gr.HTML("<h3>Upload Document</h3>")
            file_input = gr.File(
                label="Select PDF or DOCX file",
                file_types=[".pdf", ".docx"],
                file_count="single"
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                lines=2,
                value="No file uploaded yet..."
            )

            
            gr.HTML("<h3>Document Info</h3>")
            doc_info = gr.Markdown("No document loaded")
            

        with gr.Column(scale=2):
            gr.HTML("<h3>Chat with Document</h3>")
            chatbot = gr.Chatbot(
                height=450,
                show_label=False
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

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
        debug=True,
        share=True,
        inbrowser=True
    )
