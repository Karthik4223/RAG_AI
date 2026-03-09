# Production-Ready RAG System with Google Gemini

A robust, modular, and scalable Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and Google Gemini.

## 🚀 Features

- **Gemini Powered**: Uses `gemini-2.5-flash` for high-speed generation and `text-embedding-004` (or `gemini-embedding-001`) for embeddings.
- **Semantic Chunking**: Intelligent document splitting that understands content meaning rather than just character counts.
- **FastAPI Backend**: High-performance asynchronous API with automatic Swagger documentation.
- **ChromaDB Integration**: Persistent vector storage for fast semantic similarity search.
- **Hybrid Ingestion**: Supports PDF, DOCX, and TXT file uploads, as well as direct manual text ingestion.
- **Grounded Prompting**: Strict system prompts to prevent AI hallucination by ensuring answers are exclusively based on retrieved context.
- **Similarity Thresholding**: Configurable precision to ensure only highly relevant data is used for answers.

## 🛠️ Architecture

1. **Ingestion**: Documents are parsed, semantically chunked, and embedded via Gemini API.
2. **Storage**: Vectors and metadata are stored in a persistent ChromaDB instance.
3. **Retrieval**: User queries are embedded and compared against the vector store using cosine similarity.
4. **Generation**: Top-K relevant chunks are injected into a specialized prompt for Gemini 2.5 Flash to generate a factual answer.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file based on `.env.example`:
   ```env
   GOOGLE_API_KEY=your_key_here
   GEMINI_MODEL=models/gemini-2.5-flash
   GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
   ```

## 🏃 Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API documentation at: `http://localhost:8000/docs`

## 📡 API Endpoints

- `POST /api/v1/upload`: Upload and index document files (PDF, DOCX, TXT).
- `POST /api/v1/ingest-text`: Manually add text snippets with custom titles and metadata.
- `POST /api/v1/query`: Ask questions based on your indexed data.
- `POST /api/v1/feedback`: Submit user feedback for model improvement.

## 🔒 Security & Privacy

- Uses `.env` for secrets management.
- Local vector storage (ChromaDB) ensures your document metadata stays on-premise.
- Grounded prompts minimize "hallucination" risks in production environments.

---
Built with ❤️ using Google Gemini and FastAPI.
