import os
import sys
import fitz  # PyMuPDF
from docx import Document
import re
from typing import List, Dict, Optional
import json
from pathlib import Path
import logging
import uuid
import numpy as np
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_input(prompt: str = "") -> str:
    """Safe input handler that catches EOF and KeyboardInterrupt"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return "quit"


class DocumentProcessor:
    """
    Class to handle document input and text extraction from PDF and Word documents
    """
    
    def __init__(self):
        # Only support PDF and DOCX to avoid compatibility issues
        self.supported_formats = ['.pdf', '.docx']
        self.extracted_documents = []
    
    def get_document_input(self) -> List[str]:
        """
        Get input for document extraction - allows user to specify document paths
        """
        print("=== Document-Based RAG Chatbot - Document Input ===")
        print("Supported formats: PDF (.pdf), Word (.docx)")
        print("\nOptions:")
        print("1. Enter individual document paths")
        print("2. Enter directory path (will process all supported documents)")
        print("3. Use sample documents (for testing)")
        
        choice = safe_input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            return self._get_individual_paths()
        elif choice == "2":
            return self._get_directory_path()
        elif choice == "3":
            return self._create_sample_documents()
        else:
            print("Invalid choice. Please provide document paths manually.")
            return self._get_individual_paths()
    
    def _get_individual_paths(self) -> List[str]:
        """Get individual document paths from user"""
        document_paths = []
        print("\nEnter document paths (press Enter with empty line to finish):")
        
        while True:
            path = safe_input("Document path: ").strip()
            if not path or path.lower() == 'quit':
                break
            
            if os.path.exists(path) and any(path.lower().endswith(ext) for ext in self.supported_formats):
                document_paths.append(path)
                print(f"✓ Added: {path}")
            else:
                print(f"✗ Invalid path or unsupported format: {path}")
        
        return document_paths
    
    def _get_directory_path(self) -> List[str]:
        """Get all supported documents from a directory"""
        dir_path = safe_input("Enter directory path: ").strip()
        
        if not dir_path or dir_path.lower() == 'quit':
            return []
        
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"Invalid directory path: {dir_path}")
            return []
        
        document_paths = []
        for file_path in Path(dir_path).rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                document_paths.append(str(file_path))
        
        print(f"Found {len(document_paths)} supported documents in directory")
        return document_paths
    
    def _create_sample_documents(self) -> List[str]:
        """Create sample documents for testing"""
        print("Creating sample documents for testing...")
        
        # Create sample directory
        sample_dir = Path("sample_documents")
        sample_dir.mkdir(exist_ok=True)
        
        # Sample document content
        sample_texts = [
            {
                "filename": "artificial_intelligence.txt",
                "content": """
                Artificial Intelligence: An Overview
                
                Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
                that are programmed to think and learn like humans. The term may also be applied to any 
                machine that exhibits traits associated with a human mind such as learning and problem-solving.
                
                Types of AI:
                1. Narrow AI - AI that is designed to perform a narrow task
                2. General AI - AI that can perform any intellectual task that a human can
                3. Super AI - AI that surpasses human intelligence
                
                Applications of AI include machine learning, natural language processing, robotics, 
                and computer vision. AI is transforming industries such as healthcare, finance, 
                transportation, and entertainment.
                """
            },
            {
                "filename": "machine_learning.txt", 
                "content": """
                Machine Learning Fundamentals
                
                Machine Learning (ML) is a subset of artificial intelligence that provides systems 
                the ability to automatically learn and improve from experience without being explicitly programmed.
                
                Types of Machine Learning:
                
                1. Supervised Learning
                   - Uses labeled training data
                   - Examples: classification, regression
                   
                2. Unsupervised Learning
                   - Finds hidden patterns in data without labels
                   - Examples: clustering, association rules
                   
                3. Reinforcement Learning
                   - Learns through interaction with environment
                   - Uses rewards and punishments
                
                Popular algorithms include decision trees, neural networks, support vector machines,
                and random forests. Python libraries like scikit-learn, TensorFlow, and PyTorch
                are commonly used for machine learning implementations.
                """
            }
        ]
        
        created_files = []
        for doc in sample_texts:
            file_path = sample_dir / doc["filename"]
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc["content"])
                created_files.append(str(file_path))
                print(f"✓ Created: {file_path}")
            except Exception as e:
                logger.error(f"Error creating sample file {file_path}: {e}")
        
        return created_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF file with page information"""
        try:
            doc = fitz.open(pdf_path)
            extracted_pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")  # Use "text" method for cleaner extraction
                
                if text.strip():
                    extracted_pages.append({
                        'filename': os.path.basename(pdf_path),
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'file_path': pdf_path
                    })
            
            doc.close()
            logger.info(f"Extracted {len(extracted_pages)} pages from {pdf_path}")
            return extracted_pages
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return []
    
    def extract_text_from_docx(self, docx_path: str) -> List[Dict]:
        """Extract text from Word document"""
        try:
            doc = Document(docx_path)
            extracted_content = []
            
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            if full_text:
                extracted_content.append({
                    'filename': os.path.basename(docx_path),
                    'page_number': 1,
                    'text': '\n\n'.join(full_text),
                    'file_path': docx_path
                })
            
            logger.info(f"Extracted content from {docx_path}")
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error extracting DOCX {docx_path}: {e}")
            return []
    
    def extract_text_from_txt(self, txt_path: str) -> List[Dict]:
        """Extract text from plain text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                return [{
                    'filename': os.path.basename(txt_path),
                    'page_number': 1,
                    'text': text,
                    'file_path': txt_path
                }]
            
            logger.info(f"Extracted content from {txt_path}")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting TXT {txt_path}: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and formatting artifacts"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def process_documents(self, document_paths: List[str]) -> List[Dict]:
        """Process all documents and extract text with metadata"""
        if not document_paths:
            print("No documents to process.")
            return []
        
        all_extracted_content = []
        
        for doc_path in document_paths:
            print(f"Processing: {doc_path}")
            
            if doc_path.lower().endswith('.pdf'):
                content = self.extract_text_from_pdf(doc_path)
            elif doc_path.lower().endswith('.docx'):
                content = self.extract_text_from_docx(doc_path)
            elif doc_path.lower().endswith('.txt'):
                content = self.extract_text_from_txt(doc_path)
            else:
                logger.warning(f"Unsupported file format: {doc_path}")
                continue
            
            # Clean the text in each extracted content
            for item in content:
                item['text'] = self.clean_text(item['text'])
            
            all_extracted_content.extend(content)
        
        self.extracted_documents = all_extracted_content
        print(f"\n✓ Successfully processed {len(document_paths)} documents")
        print(f"✓ Extracted {len(all_extracted_content)} content sections")
        
        return all_extracted_content


class TextChunker:
    """
    Class to handle text chunking with overlapping sliding window strategy
    """
    
    def __init__(self, chunk_size: int = 500, overlap_size: int = 50):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Number of words per chunk
            overlap_size: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = min(overlap_size, chunk_size - 1)  # Ensure overlap is less than chunk_size
        print(f"TextChunker initialized with chunk_size={chunk_size}, overlap_size={self.overlap_size}")
    
    def split_into_words(self, text: str) -> List[str]:
        """Split text into words using regex"""
        words = re.findall(r'\S+', text)
        return words
    
    def create_chunks_with_overlap(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Create overlapping chunks from text
        
        Args:
            text: Input text to chunk
            metadata: Original document metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        words = self.split_into_words(text)
        
        if len(words) <= self.chunk_size:
            return [{
                'chunk_id': 0,
                'text': text,
                'word_count': len(words),
                'filename': metadata.get('filename', 'unknown'),
                'page_number': metadata.get('page_number', 1),
                'file_path': metadata.get('file_path', ''),
                'chunk_start_word': 0,
                'chunk_end_word': len(words)
            }]
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunk_data = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'word_count': len(chunk_words),
                'filename': metadata.get('filename', 'unknown'),
                'page_number': metadata.get('page_number', 1),
                'file_path': metadata.get('file_path', ''),
                'chunk_start_word': start_idx,
                'chunk_end_word': end_idx
            }
            
            chunks.append(chunk_data)
            
            if end_idx >= len(words):
                break
            
            # Ensure start_idx is always positive and moves forward
            start_idx = max(start_idx + 1, end_idx - self.overlap_size)
            chunk_id += 1
        
        return chunks
    
    def process_extracted_documents(self, extracted_documents: List[Dict]) -> List[Dict]:
        """Process all extracted documents and create chunks"""
        all_chunks = []
        
        for doc_idx, doc_content in enumerate(extracted_documents):
            print(f"Chunking document {doc_idx + 1}/{len(extracted_documents)}: {doc_content['filename']}")
            
            doc_chunks = self.create_chunks_with_overlap(
                text=doc_content['text'],
                metadata=doc_content
            )
            
            for chunk in doc_chunks:
                chunk['document_index'] = doc_idx
                chunk['total_chunks_in_doc'] = len(doc_chunks)
            
            all_chunks.extend(doc_chunks)
            print(f"  ✓ Created {len(doc_chunks)} chunks")
        
        print(f"\n✓ Total chunks created: {len(all_chunks)}")
        return all_chunks


class EmbeddingManager:
    """
    Manages embedding generation and vector storage operations
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding manager with sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model.eval()  # Set to evaluation mode
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
        
        # Initialize Qdrant client
        try:
            self.vector_client = QdrantClient(path="./qdrant_data")
            self.collection_name = "document_chunks"
            
            # Create collection
            self.vector_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            print(f"Qdrant collection '{self.collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise
        
        print(f"EmbeddingManager initialized with {model_name}, dim={self.embedding_dim}")
    
    def process_chunks_to_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Convert text chunks to embeddings and store in vector database"""
        if not chunks:
            print("No chunks to process")
            return []
        
        print(f"Processing {len(chunks)} chunks to embeddings...")
        
        # Extract text from chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches to avoid memory issues
        try:
            embeddings = self.embedding_model.encode(
                chunk_texts, 
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            print(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
        
        # Prepare points for Qdrant
        points = []
        enhanced_chunks = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Use a more compatible ID format
            point_id = int(uuid.uuid4().int % (2**63 - 1))  # Ensure 64-bit signed integer
            
            try:
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'filename': chunk['filename'],
                        'page_number': chunk['page_number'],
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'word_count': chunk['word_count'],
                        'file_path': chunk.get('file_path', ''),
                        'document_index': chunk.get('document_index', 0)
                    }
                )
                
                points.append(point)
                
                enhanced_chunk = chunk.copy()
                enhanced_chunk['vector_id'] = point_id
                enhanced_chunk['embedding_dim'] = self.embedding_dim
                enhanced_chunks.append(enhanced_chunk)
                
            except Exception as e:
                logger.error(f"Error creating point for chunk {i}: {e}")
                continue
        
        # Store in Qdrant
        try:
            self.vector_client.upsert(self.collection_name, points)
            print(f"✓ Stored {len(points)} embeddings in vector database")
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {e}")
            return []
        
        return enhanced_chunks
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks based on query"""
        try:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            
            search_results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            print(f"Found {len(search_results)} relevant chunks for query")
            
            relevant_chunks = []
            for result in search_results:
                chunk_data = result.payload.copy()
                chunk_data['similarity_score'] = result.score
                chunk_data['vector_id'] = result.id
                relevant_chunks.append(chunk_data)
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []


class LLMAnswerGenerator:
    """
    Handles answer generation using open-source LLM
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize LLM for answer generation"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model_name = model_name
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set padding side for generation
            self.tokenizer.padding_side = "left"
            
            print(f"LLM initialized: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.model = None
            self.tokenizer = None
            self.model_name = "fallback"
        
        self.max_context_length = 1800  # Conservative limit for context
    
    def truncate_context(self, context: str, question: str) -> str:
        """Truncate context to fit within model limits"""
        if not self.tokenizer:
            return context[:2000]  # Fallback truncation
        
        try:
            # Tokenize context and question separately
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
            
            # Reserve tokens for prompt structure and generation
            available_tokens = self.max_context_length - len(question_tokens) - 100
            
            if len(context_tokens) > available_tokens:
                # Keep the most recent context tokens
                context_tokens = context_tokens[-available_tokens:]
            
            return self.tokenizer.decode(context_tokens, skip_special_tokens=True)
        except:
            # Fallback to character-based truncation
            return context[:5000]
    
    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer based on context and question
        """
        if not self.model or not self.tokenizer:
            # Fallback to simple context extraction
            return self._fallback_answer(context, question)
        
        # Truncate context to fit model limits
        truncated_context = self.truncate_context(context, question)
        
        # Construct prompt
        prompt = f"""Based on the following context, answer the question. Cite the document name and page number.

Context:
{truncated_context}

Question: {question}

Answer:"""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_context_length,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=550,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs['attention_mask']
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()
            
            return answer if answer else "No answer found in the provided context."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._fallback_answer(context, question)
    
    def _fallback_answer(self, context: str, question: str) -> str:
        """Fallback method when LLM is not available"""
        # Simple keyword matching as fallback
        question_words = set(question.lower().split())
        context_sentences = context.split('.')
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if len(question_words.intersection(sentence_words)) > 1:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return " ".join(relevant_sentences[:3]) + "..."
        else:
            return "No answer found in the provided context."


class RAGChatbot:
    """
    Complete RAG Chatbot that combines all components
    """
    
    def __init__(self):
        """Initialize RAG Chatbot with all components"""
        print("Initializing RAG Chatbot components...")
        
        try:
            self.document_processor = DocumentProcessor()
            self.chunker = TextChunker(chunk_size=500, overlap_size=50)
            self.embedding_manager = EmbeddingManager()
            self.llm = LLMAnswerGenerator()
            self.chat_history = []
            
            print("✓ RAGChatbot initialized with all components")
        except Exception as e:
            logger.error(f"Error initializing RAG Chatbot: {e}")
            raise
    
    def setup_documents(self):
        """Setup documents for the chatbot"""
        try:
            # Get document input
            document_paths = self.document_processor.get_document_input()
            
            if not document_paths:
                print("No documents provided. Exiting.")
                sys.exit(0)
            
            # Process documents
            extracted_content = self.document_processor.process_documents(document_paths)
            
            if not extracted_content:
                print("No content extracted from documents. Exiting.")
                sys.exit(0)
            
            # Create chunks
            chunks = self.chunker.process_extracted_documents(extracted_content)
            
            if not chunks:
                print("No chunks created. Exiting.")
                sys.exit(0)
            
            # Generate embeddings and store
            self.enhanced_chunks = self.embedding_manager.process_chunks_to_embeddings(chunks)
            
            if not self.enhanced_chunks:
                print("No embeddings generated. Exiting.")
                sys.exit(0)
            
            print(f"✓ System ready with {len(self.enhanced_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error setting up documents: {e}")
            sys.exit(1)
    
    def process_query(self, user_query: str, top_k: int = 3) -> Dict:
        """Process user query and generate answer"""
        start_time = datetime.now()
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.embedding_manager.search_relevant_chunks(user_query, top_k)
            
            if not relevant_chunks:
                return {
                    'answer': "No relevant information found in the documents.",
                    'sources': [],
                    'query': user_query,
                    'response_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Prepare context for LLM
            context_parts = []
            sources = []
            
            for chunk in relevant_chunks:
                context_parts.append(f"Document: {chunk['filename']}, Page: {chunk['page_number']}\n{chunk['text']}")
                sources.append({
                    'filename': chunk['filename'],
                    'page_number': chunk['page_number'],
                    'chunk_id': chunk['chunk_id'],
                    'similarity_score': chunk.get('similarity_score', 0.0)
                })
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate answer using LLM
            answer = self.llm.generate_answer(context, user_query)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                'answer': answer,
                'sources': sources,
                'query': user_query,
                'response_time': response_time,
                'retrieved_chunks': len(relevant_chunks)
            }
            
            self.chat_history.append(response)
            
            # Ensure response time is under 15 seconds
            if response_time > 15:
                print(f"Warning: Response time {response_time:.2f}s exceeds 15s limit")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': "Error processing your query. Please try again.",
                'sources': [],
                'query': user_query,
                'response_time': (datetime.now() - start_time).total_seconds()
            }
    
    def display_response(self, response: Dict):
        """Display chatbot response in formatted way"""
        print(f"\nQuery: {response['query']}")
        print("=" * 50)
        print(f"Answer: {response['answer']}")
        print(f"\nResponse Time: {response['response_time']:.2f} seconds")
        print("\nSources:")
        for i, source in enumerate(response['sources'], 1):
            similarity_score = source.get('similarity_score', 0.0)
            print(f"  {i}. {source['filename']} - Page {source['page_number']}, Chunk {source['chunk_id']}")
            print(f"     Similarity: {similarity_score:.3f}")
    
    def start_interactive_session(self):
        """Start interactive chat session"""
        print("\n" + "=" * 60)
        print("DOCUMENT-BASED RAG CHATBOT")
        print("=" * 60)
        print("Ask questions about the loaded documents.")
        print("Type 'quit' to exit, 'history' to see chat history.")
        print("-" * 60)
        
        while True:
            try:
                user_input = safe_input("\nYour question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'history':
                    if self.chat_history:
                        print(f"\nChat History ({len(self.chat_history)} queries):")
                        for i, item in enumerate(self.chat_history, 1):
                            print(f"  {i}. {item['query']}")
                    else:
                        print("\nNo chat history yet.")
                    continue
                
                response = self.process_query(user_input)
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print("An error occurred. Please try again.")


# Main execution
def main():
    """Main function to run the RAG chatbot"""
    try:
        print("Document-Based RAG Chatbot")
        print("=" * 50)
        
        # Initialize chatbot
        chatbot = RAGChatbot()
        
        # Setup documents
        chatbot.setup_documents()
        
        # Start interactive session
        chatbot.start_interactive_session()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("Application encountered a fatal error. Please check the logs.")


if __name__ == "__main__":
    main()
