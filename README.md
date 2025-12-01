# 📚 Multimodal RAG - Universal Document Q&A
Universal document Q&A powered by multimodal RAG. Upload any file format (PDFs with images, DOCX, PPTX, code, etc.) and chat with your documents using Groq's lightning-fast LLMs.
## ✨ Features
- **📄 Multi-format support**: PDF, DOCX, PPTX, XLSX, TXT, Markdown, JSON, XML, HTML, YAML, **20+ code formats**
- **🖼️ Multimodal RAG**: CLIP embeddings for PDFs with images + text
- **⚡ Text RAG**: sentence-transformers for all other files (no version conflicts)
- **🤖 Multiple LLMs**: GPT-OSS 120B, Llama 3.1 8B, Qwen3 32B per query
- **📱 Streamlit UI**: Chat history, retrieved context expanders, model badges
- **🔍 Smart retrieval**: Shows relevant text chunks + image locations
- **🚀 Cross-platform**: Windows/Mac/Linux with temp file handling

## 🚀 Quick Start
```bash
# 1. Clone & install
git clone <your-repo>
cd multimodal-rag
pip install -r requirements.txt

# 2. Setup Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. Run
streamlit run app.py
```

## 📁 Supported Formats
| Format | Icon | Processor |
|--------|------|-----------|
| PDF | 📄 | CLIP + Text (multimodal) |
| DOCX | 📝 | Paragraph extraction |
| PPTX | 🎬 | Slide text extraction |
| XLSX | 📊 | Cell text extraction |
| TXT/MD | 📋 | Chunked text |
| Python | 🐍 | Code-aware chunking |
| JS/TS/Java | 📜 | Code-aware chunking |
| C++/C/Go | ⚙️ | Code-aware chunking |
| JSON/XML | 🔧 | Structured text |
| HTML/YAML | 🌐 | Markup parsing |

## 🏗️ Architecture
**Two processing paths:**
```
PDF Files → CLIP embeddings → multimodal_groq.py → FAISS (text+images)
Non-PDF → sentence-transformers → FAISS (text-only)
```

## 📦 Installation
<details>
<summary>Complete setup (click to expand)</summary>

```bash
# Core dependencies
pip install streamlit langchain langchain-groq langchain-community sentence-transformers faiss-cpu

# File format support (optional)
pip install python-docx python-pptx openpyxl

# Multimodal PDF support (separate file)
pip install multimodal-groq  # or copy multimodal_groq.py

# Development
pip install python-dotenv
```

</details>

**requirements.txt:**
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-groq>=0.1.0
langchain-community>=0.0.20
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
python-docx>=0.8.11
python-pptx>=0.6.21
openpyxl>=3.1.0
python-dotenv>=1.0.0
```

## 🎮 Usage
### 1. **Upload File** (Sidebar)
```
📁 Upload: PDF/DOCX/PPTX/TXT/Code/etc.
✅ "Processed! (42 chunks)"
```

### 2. **Query Document**
```
💬 "What are the main findings?"
⚙️ Model: Llama 3.1 8B
📊 Chunks: 5
```

### 3. **View Results**
```
🤖 Response: "The document discusses..."
📋 Retrieved Context:
  📖 Text from Page 3 (chunk 1) [expander]
  🖼️ Image from Page 5
```

### 4. **Chat History**
```
Q1: Summary? → Llama 3.1 [repeat/clear]
```

## 🔧 File Processing Paths
| File Type | Embeddings | Retrieval | Message |
|-----------|------------|-----------|---------|
| **PDF** | CLIP (multimodal) | `retrieve_multimodal()` | `create_multimodal_message()` |
| **Others** | all-MiniLM-L6-v2 | `similarity_search()` | `create_text_message()` |

## 🐛 Troubleshooting
<details>
<summary>Common Issues (click to expand)</summary>

| Error | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Create `.env` with `GROQ_API_KEY=gsk_...` |
| `ImportError: ModelProfile` | Use `sentence-transformers` (fixed in this repo) |
| `'SentenceTransformerEmbeddings' not callable` | Updated with `Embeddings` base class |
| `multimodal_groq.py not found` | Copy from original repo or disable PDF |
| `python-docx not installed` | `pip install python-docx` |

</details>

## 📊 Dependencies Overview
```
Core: streamlit + langchain + groq
Text: sentence-transformers + FAISS
Files: docx/pptx/openpyxl
PDF: multimodal_groq + CLIP
```

## 🤝 Contributing
1. Fork the repo
2. Create feature branch (`git checkout -b feature/add-format`)
3. Commit changes (`git commit -m 'Add YAML support'`)
4. Push (`git push origin feature/add-format`)
5. Open Pull Request

**New formats?** Add to `process_file()` + `SUPPORTED_FORMATS`!

## 📄 License
MIT License - see [LICENSE](LICENSE) © 2025

***

**Built with ❤️ for document AI enthusiasts** | [Star ⭐](https://github.com/stargazers) | [Issues](https://github.com/issues)
