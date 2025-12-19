# -*- coding: utf-8 -*-
"""
Multimodal RAG Module: CLIP + FAISS + Groq Integration

Handles:
- PDF processing with text + image extraction
- CLIP embeddings for multimodal retrieval
- FAISS vector store management
- Multimodal message creation for LLM
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import tempfile
import torch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF

try:
    from PIL import Image
    import io
except ImportError:
    Image = None
    io = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor = None
    CLIPModel = None

# -------------------------------------------------------------------
# CLIP Model Initialization
# -------------------------------------------------------------------

def init_clip_model() -> Tuple:
    """
    Initialize CLIP model and processor for multimodal embeddings.
    
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: CLIP model and processor
    """
    if CLIPModel is None or CLIPProcessor is None:
        raise ImportError(
            "Transformers library not installed. "
            "Run: pip install transformers pillow torch"
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return model, processor


# -------------------------------------------------------------------
# PDF Processing with Text + Images
# -------------------------------------------------------------------

def process_pdf(
    file_path: str,
    clip_model: Any,
    clip_processor: Any
) -> Tuple[List[Document], np.ndarray, Dict]:
    """
    Process PDF file to extract text and images with CLIP embeddings.
    
    Args:
        file_path: Path to PDF file
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
    
    Returns:
        Tuple of (all_docs, embeddings_array, image_data_store)
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    try:
        pdf_document = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {str(e)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for page_idx, page in enumerate(pdf_document):
        # Extract text from page
        text = page.get_text()
        
        if text.strip():
            # Create document for page text
            doc = Document(
                page_content=text,
                metadata={
                    "page": page_idx,
                    "type": "text",
                    "source": "pdf"
                }
            )
            all_docs.append(doc)
            
            # Get CLIP embedding for text
            with torch.no_grad():
                text_inputs = clip_processor(
                    text=text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=77
                ).to(device)
                text_embedding = clip_model.get_text_features(**text_inputs)
                text_embedding = text_embedding.cpu().numpy()[0]
                all_embeddings.append(text_embedding)
        
        # Extract images from page
        image_list = page.get_images(full_info=True)
        
        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                else:  # CMYK
                    img = Image.new("RGB", (pix.width, pix.height))
                    img.frombytes(pix.tobytes("rgb"))
                
                # Get CLIP embedding for image
                with torch.no_grad():
                    image_inputs = clip_processor(
                        images=img,
                        return_tensors="pt"
                    ).to(device)
                    image_embedding = clip_model.get_image_features(**image_inputs)
                    image_embedding = image_embedding.cpu().numpy()[0]
                    all_embeddings.append(image_embedding)
                
                # Store image metadata
                image_id = f"page_{page_idx}_img_{img_idx}"
                image_data_store[image_id] = {
                    "page": page_idx,
                    "xref": xref,
                    "width": pix.width,
                    "height": pix.height
                }
                
                # Create document for image reference
                img_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={
                        "page": page_idx,
                        "type": "image",
                        "image_id": image_id,
                        "source": "pdf"
                    }
                )
                all_docs.append(img_doc)
                
            except Exception as e:
                print(f"Warning: Failed to process image on page {page_idx}: {str(e)}")
                continue
    
    pdf_document.close()
    
    # Convert embeddings to numpy array
    if all_embeddings:
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
    else:
        # Fallback: create dummy embeddings if none extracted
        embeddings_array = np.zeros((len(all_docs), 512), dtype=np.float32)
    
    return all_docs, embeddings_array, image_data_store


# -------------------------------------------------------------------
# Vector Store Management
# -------------------------------------------------------------------

def build_vector_store(all_docs: List[Document], embeddings_array: np.ndarray) -> Any:
    """
    Build FAISS vector store from documents and embeddings.
    
    Args:
        all_docs: List of LangChain Document objects
        embeddings_array: Numpy array of embeddings (n_docs x embedding_dim)
    
    Returns:
        FAISS vector store instance
    """
    
    if len(all_docs) == 0:
        raise ValueError("No documents to build vector store from")
    
    if embeddings_array.shape[0] != len(all_docs):
        raise ValueError(
            f"Embeddings count ({embeddings_array.shape[0]}) "
            f"doesn't match documents count ({len(all_docs)})"
        )
    
    # Create custom embeddings wrapper for FAISS compatibility
    class NumpyEmbeddings:
        def __init__(self, embeddings_array):
            self.embeddings = embeddings_array
            self.embedding_dim = embeddings_array.shape[1]
        
        def embed_documents(self, texts):
            """Return precomputed embeddings - not used in this flow."""
            return self.embeddings.tolist()
        
        def embed_query(self, text):
            """Return zero vector for queries - will use similarity search."""
            return np.zeros(self.embedding_dim).tolist()
    
    embeddings_model = NumpyEmbeddings(embeddings_array)
    
    try:
        # Create FAISS store from documents
        vector_store = FAISS.from_documents(all_docs, embeddings_model)
        return vector_store
    except Exception as e:
        raise ValueError(f"Failed to build vector store: {str(e)}")


# -------------------------------------------------------------------
# Multimodal Retrieval
# -------------------------------------------------------------------

def retrieve_multimodal(
    query: str,
    vector_store: Any,
    clip_model: Any,
    clip_processor: Any,
    k: int = 5
) -> List[Document]:
    """
    Retrieve documents using multimodal similarity search.
    
    Args:
        query: Query string
        vector_store: FAISS vector store
        clip_model: CLIP model instance
        clip_processor: CLIP processor instance
        k: Number of results to retrieve
    
    Returns:
        List of relevant Document objects
    """
    
    if vector_store is None:
        raise ValueError("Vector store not initialized")
    
    try:
        # Get CLIP embedding for query
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with torch.no_grad():
            query_inputs = clip_processor(
                text=query,
                return_tensors="pt",
                truncation=True,
                max_length=77
            ).to(device)
            query_embedding = clip_model.get_text_features(**query_inputs)
            query_embedding = query_embedding.cpu().numpy()[0]
        
        # Similarity search in FAISS
        docs = vector_store.similarity_search_by_vector(query_embedding, k=k)
        
        return docs
    
    except Exception as e:
        raise ValueError(f"Retrieval failed: {str(e)}")


# -------------------------------------------------------------------
# Message Creation
# -------------------------------------------------------------------

def create_multimodal_message(
    query: str,
    context_docs: List[Document]
) -> HumanMessage:
    """
    Create a multimodal message for LLM from retrieved documents.
    
    Args:
        query: User query
        context_docs: List of retrieved Document objects
    
    Returns:
        HumanMessage for LLM invocation
    """
    
    # Separate text and image documents
    text_content = []
    image_content = []
    
    for doc in context_docs:
        if doc.metadata.get("type") == "text":
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            text_content.append(
                f"[{source.upper()} - Page {page}]\n{doc.page_content}"
            )
        elif doc.metadata.get("type") == "image":
            image_id = doc.metadata.get("image_id", "unknown")
            page = doc.metadata.get("page", "?")
            image_content.append(f"[Image on page {page}: {image_id}]")
    
    # Build message content
    context_text = "\n\n".join(text_content)
    
    message_content = f"""You are a helpful assistant analyzing documents with text and images.

Based on the following context from the document:

{context_text}
"""
    
    if image_content:
        message_content += f"\nImages in document: {', '.join(image_content)}\n"
    
    message_content += f"""
Please answer this question: {query}

Provide a clear, concise answer based on the context provided. If images are relevant, mention their locations."""
    
    return HumanMessage(content=message_content)


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def get_embeddings_dimension(clip_model: Any) -> int:
    """Get the embedding dimension from CLIP model."""
    return 512  # CLIP base model uses 512-dim embeddings


def validate_pdf(file_path: str) -> bool:
    """Validate if file is a valid PDF."""
    try:
        doc = fitz.open(file_path)
        doc.close()
        return True
    except Exception:
        return False
