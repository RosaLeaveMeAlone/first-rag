# RAG with ChromaDB

RAG (Retrieval-Augmented Generation) system that allows you to upload PDF documents and ask questions about their content using artificial intelligence.

## 🛠️ Technologies

- **LangChain** - Framework for AI applications
- **ChromaDB** - Persistent vector database
- **Streamlit** - Interactive web interface
- **OpenAI** - Embeddings and chat completions
- **Docker Compose** - Container orchestration

## 🚀 Installation

1. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

2. **Run with Docker:**
```bash
docker-compose up -d
```

3. **Access the application:**
- Main interface: http://localhost:8501
- ChromaDB health status: http://localhost:8000/api/v2/heartbeat

## 💡 Usage

1. Upload PDF documents from the web interface
2. Ask questions about the content
3. Documents are stored persistently in ChromaDB
4. Responses include the sources used

## ⚙️ Configuration

Main variables in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
CHROMA_HOST=chromadb
CHROMA_PORT=8000
COLLECTION_NAME=rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Document Processing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Maximum characters per text chunk for processing |
| `CHUNK_OVERLAP` | 200 | Character overlap between consecutive chunks to maintain context |

## 🛠️ Docker Commands

```bash
# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down
```