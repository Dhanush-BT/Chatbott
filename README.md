
<p align="center">
  <a href="https://your-project-url.com">
    <img src="documind.jpeg" alt="DocuMate Header" width="300" />
  </a>
</p>

<p align="center">
  <a href="https://your-project-url.com/demo">Demo</a> ·
  <a href="https://your-project-url.com/docs">Documentation</a> ·
  <a href="https://your-project-url.com/examples">Examples</a> ·
  <a href="https://discord.gg/your-community">Community</a> ·
  <a href="mailto:your@email.com">Contact Us</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Multilingual-Supported-brightgreen" alt="Language Support Badge" />
  <img src="https://img.shields.io/badge/LLM-Gemma_2B-blueviolet" alt="LLM Badge" />
  <img src="https://img.shields.io/badge/Interactive_UI-Yes-orange" alt="UI Badge" />
  <img src="https://img.shields.io/badge/Offline-Yes-brightgreen" alt="Offline Badge" />
</p>

# 📄 **DocuMind**
### *Ask your documents. Multilingual. Open-source. Offline-ready.*

DocuMind is an open-source, RAG-based chatbot that allows you to interact with your documents in **natural language**. Upload `.pdf` or `.docx` files, and ask questions — the chatbot answers with references to the **exact source**, even if it's in **English, Tamil, Malayalam, Japanese**, or any other languages.

---

## 🚀 Features

- 🔎 Document-based Retrieval-Augmented Generation
- 📁 Upload `.pdf`, `.docx`, or entire folders
- 💬 Multilingual input support (Tamil, Malayalam, Japanese, English, etc)
- 🧠 Powered by Gemma-2B (offline-compatible)
- 💡 Source-cited answers (Filename, Chunk used, Response time, Chunk length, CPU usage, Memory usage)
- 🖼️ Interactive Gradio UI
- 🤖 Greeting & casual conversation support

---

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/Dhanush-BT/Chatbott.git
cd DocuMate

# Install dependencies
pip install -r requirements.txt

# Run in terminal mode
python main.py

# OR Run with Gradio UI
python ui.py
```
---

## 💡 Usage

- Ask any **factual question** based on the uploaded documents.
- The chatbot responds using only the retrieved chunks from documents.
- It also displays the **exact source**: Filename, Chunk used, Response time, Chunk length, CPU usage, Memory usage.
- If no answer is found, the chatbot explicitly says so.
- Additional features: multilingual queries, greeting support, and a rich UI.

---

## 🧱 Architectural Decisions

- **Modular Design**: Separated into components:
  - `DocumentProcessor`: Extraction from PDFs/DOCX
  - `TextChunker`: Token-based sliding chunk creation
  - `EmbeddingManager`: Embedding + Qdrant storage
  - `LLMAnswerGenerator`: Offline LLM-based generation
  - `RAGChatbot`: Combines all for unified interface

- **Offline Support**: No API calls; runs entirely offline with open-source models.
- **Fast Retrieval**: Designed to meet the strict 15-second response limit.

---

## 🔍 Observations

- Embedding quality greatly influences response relevance.
- Overlapping chunk strategy improved chunk completeness and reduced partial answers.
- Gradio UI made debugging easier during testing.
- Token size of LLM had to be managed to avoid memory spikes on T4.

---

## ✂️ Chunking Strategy

To ensure efficient document understanding and minimize context overflow, the chatbot uses a **sliding window chunking strategy** based on character length:

- **Max Chunk Size**: 3,500 characters  
- **Overlap Size**: 200 characters  
- **Logic**: The document is split into overlapping text chunks, ensuring that contextual continuity is preserved between segments.

This method helps the LLM (Gemma 2B) understand the flow of content and avoids missing important information that could be split across chunk boundaries.

---

## 🔍 Retrieval Strategy

- The chatbot follows a lightweight, sequential retrieval strategy:

- **Iterative Chunk Search**: `Each text chunk is passed to the LLM along with the user’s query.`
- **Vector DB**: **Qdrant** (local setup)
- **Early Exit Optimization**: `As soon as the model returns a meaningful answer that doesn’t include phrases like "The document does not contain that information", the system immediately stops and returns that response to save time and computation `
- **Fallback Handling**: `If none of the chunks return a relevant answer, a default fallback response is returned: "The answer was not found in the document. `
- **Multilingual Support**: `The retrieval prompt and question are language-agnostic, allowing users to ask queries in multiple languages including English, Tamil, Malayalam, Japanese, etc.`
  
---

## 🧠 LLM Answer Generation (Gemma 2B)

- **Model Used**: `gemma-2b-it` (instruction-tuned)
- **Why Gemma?**
  - Lightweight (2B parameters) and efficient for GPU usage
  - Supports multilingual instruction following
  - Fully open-source and suitable for offline inference
- **Prompt Format**:
  - If no answer is found, the model returns:
    “No answer found in the provided context.”

---

## 🧪 Performance on Tesla T4 (GPU/CPU Usage & Optimization)

Our solution has been carefully optimized to run fully on a **Tesla T4 GPU (16GB VRAM)**. Resource consumption and response time:

### 🚀 Response Time

| Mode                     | Average Response Time  |
|--------------------------|------------------------|
| **GPU (Tesla T4)**       | ✅ Under 12 seconds   |
| **CPU-only (fallback)**  |  ~30–40 seconds        |
-----------------------------------------------------

---

## ⚙️ Hardware Usage (Gemma-2B on Tesla T4)

| Component            | Resource Usage (VRAM) |
|----------------------|------------------------|
| Embedding Model      | ~3.2 GB                |
| Qdrant + Retrieval   | ~500 MB                |
| **Gemma-2B LLM**     | ~6.5–7.0 GB            |
| **Total**            | ~10.5–11.0 GB          |
------------------------------------------------


- ✅ Optimized to run within **16GB GPU limit (T4)**
- Quantized or float16 versions used for lower memory
- Batched embeddings and prompt truncation enabled

---

## 📌 Submission Checklist

| Requirement                              | ✅ Done? |
|------------------------------------------|----------|
| Accuracy of Answers                      | ✅       |
| Justification of Chunking Method         | ✅       |
| Response Time Within Limit               | ✅       |
| Good README file on GitHub               | ✅       |
| All dependencies offline                 | ✅       |
| Accuracy of Source Document              | ✅       |
| Extra Enhancements                       | ✅       |
-------------------------------------------------------
