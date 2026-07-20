"""
Ingestion script for the RAG pipeline.

Loads markdown documents from ./docs, splits them into chunks, embeds them
with a local sentence-transformers model (no API key needed for embeddings),
and persists the result into a local Chroma vector store.

Usage:
    python ingest.py
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Local embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} source documents from {DOCS_DIR}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        # Explicitly use cosine distance so relevance scores are bounded 0-1.
        # Chroma's default (L2) produces scores on a different scale, which broke the relevance threshold check downstream.
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Persisted vector store to {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    build_vectorstore(chunks)
    print("Ingestion complete.")