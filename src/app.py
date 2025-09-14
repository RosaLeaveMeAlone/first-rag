import streamlit as st
import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import io
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "test-token")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

st.set_page_config(page_title="RAG Persistente", page_icon="📚", layout="wide")

st.title("📚 RAG with Persistent Vector Database | RAG con Base de Datos Vectorial Persistente")
st.markdown("*RAG application with ChromaDB for persistent document storage*")
st.markdown("*Aplicación RAG con ChromaDB para almacenamiento persistente de documentos*")

# Initialize models and clients
@st.cache_resource
def init_models():
    """Initialize OpenAI model and ChromaDB client"""
    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.1
    )

    # ChromaDB client configuration
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT
    )

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    return model, chroma_client, embeddings

# RAG Prompt Template
prompt = ChatPromptTemplate.from_template("""
Responde a las preguntas basándote únicamente en el contexto proporcionado.
Proporciona la respuesta más precisa basada en la pregunta.
Si no sabes la respuesta basándote en el contexto, responde "No encuentro esa información en los documentos disponibles."

<context>
{context}
</context>

Pregunta: {input}

Respuesta:
""")

def process_uploaded_pdf(uploaded_file):
    """Process a single uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF from temporary file
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Clean up temporary file
        os.unlink(tmp_file_path)

        if not docs:
            return []

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

        # Add source metadata
        for doc in docs:
            doc.metadata['source'] = uploaded_file.name

        final_documents = text_splitter.split_documents(docs)
        return final_documents

    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        return []

def initialize_vector_store():
    """Initialize or connect to persistent vector store"""
    try:
        model, chroma_client, embeddings = init_models()

        # Check if collection exists
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            # Collection exists, connect to existing Chroma vector store
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
            st.success(f"✅ Connected to existing collection '{COLLECTION_NAME}' with {collection.count()} documents | Conectado a la colección existente '{COLLECTION_NAME}' con {collection.count()} documentos")
            return vectorstore, True

        except Exception:
            # Collection doesn't exist, need to create it
            return None, False

    except Exception as e:
        st.error(f"❌ Error conectando a ChromaDB: {str(e)}")
        return None, False

def add_documents_to_vectorstore(uploaded_files):
    """Add uploaded documents to existing vector store"""
    try:
        model, chroma_client, embeddings = init_models()

        # Get or create vector store
        try:
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
        except Exception:
            # Create new collection if it doesn't exist
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )

        all_documents = []
        processed_files = []

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                documents = process_uploaded_pdf(uploaded_file)
                if documents:
                    all_documents.extend(documents)
                    processed_files.append(uploaded_file.name)

        if not all_documents:
            st.warning("⚠️ No documents were processed successfully")
            return None

        # Add documents to vector store
        with st.spinner("🔄 Adding documents to ChromaDB..."):
            vectorstore.add_documents(all_documents)

        st.success(f"✅ Successfully added {len(all_documents)} chunks from {len(processed_files)} files:")
        for filename in processed_files:
            st.write(f"  • {filename}")

        return vectorstore

    except Exception as e:
        st.error(f"❌ Error adding documents: {str(e)}")
        return None


# Initialize the application
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.vectorstore_ready = False

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")

    # Try to connect to existing collection
    if not st.session_state.vectorstore_ready:
        vectorstore, exists = initialize_vector_store()
        if exists:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_ready = True

    # File uploader
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to your knowledge base"
    )

    if uploaded_files:
        if st.button("🚀 Process & Add Documents", type="primary"):
            vectorstore = add_documents_to_vectorstore(uploaded_files)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_ready = True
                st.rerun()

    st.divider()

    # Collection info
    st.subheader("📊 Collection Info")
    if st.button("📈 View Stats"):
        if st.session_state.vectorstore_ready:
            try:
                model, chroma_client, embeddings = init_models()
                collection = chroma_client.get_collection(COLLECTION_NAME)
                count = collection.count()
                st.info(f"📈 Documents in collection: {count}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("No collection found. Upload documents first.")

# Main interface
st.header("💬 Chat with Your Documents")

if not st.session_state.vectorstore_ready:
    st.info("🔄 Initializing connection to vector database...")
    vectorstore, exists = initialize_vector_store()
    if not exists:
        st.warning("⚠️ No existing collection found. Upload documents using the sidebar to get started.")
    else:
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_ready = True
        st.rerun()

# Chat interface
if st.session_state.vectorstore_ready:
    user_question = st.text_input("🔍 Ask a question about your documents:")

    if user_question:
        try:
            model, _, _ = init_models()

            # Create retrieval chain
            with st.spinner("🔍 Buscando información relevante..."):
                document_chain = create_stuff_documents_chain(model, prompt)
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Get response
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_question})
                response_time = time.process_time() - start_time

            # Display response
            st.markdown("### 🤖 Respuesta:")
            st.markdown(response['answer'])

            # Show processing time
            st.caption(f"⏱️ Tiempo de respuesta: {response_time:.2f} segundos")

            # Show source documents
            with st.expander("📄 Documentos fuente utilizados"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Fragmento {i+1}:**")
                    st.markdown(doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.caption(f"Fuente: {doc.metadata.get('source', 'Desconocida')}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error procesando la pregunta: {str(e)}")

else:
    st.info("🔄 Vector database is not ready. Use the sidebar to upload documents.")

# Footer
st.markdown("---")
st.markdown("*RAG Application with ChromaDB - Persistent Vector Database*")