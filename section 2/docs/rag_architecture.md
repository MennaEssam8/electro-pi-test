# RAG Pipeline Architecture Notes

## Overview
A Retrieval-Augmented Generation (RAG) system combines a retriever and a generator
to answer questions grounded in a document collection, instead of relying purely
on the LLM's parametric knowledge.

## Core Components

### 1. Document Loader
Responsible for reading raw files (PDF, Markdown, HTML) and converting them into
a unified `Document` object with `page_content` and `metadata` (source, page number).

### 2. Text Splitter
Long documents are split into smaller chunks before embedding. Common strategies:
- Fixed-size character splitting with overlap
- Recursive splitting that respects paragraph/sentence boundaries
- Semantic chunking based on embedding similarity between sentences

Chunk size is a trade-off: smaller chunks improve retrieval precision but lose
surrounding context; larger chunks preserve context but dilute relevance scores.

### 3. Embedding Model
Converts each chunk into a dense vector. The choice of embedding model affects
both retrieval quality and cost. Local models (e.g. sentence-transformers) avoid
API latency, while hosted models (OpenAI, Cohere) often perform better on
domain-specific text.

### 4. Vector Store
Stores embeddings and supports approximate nearest neighbor search. Popular
choices: FAISS (in-memory, fast, no metadata filtering out of the box), Chroma
(lightweight, persistent, good for prototyping), and Qdrant (production-grade,
supports multiple named collections and rich payload filtering).

### 5. Retriever
Given a query, embeds it and returns the top-k most similar chunks. Retrieval
quality can be improved with:
- Hybrid search (combining sparse BM25 with dense vector search)
- Re-ranking retrieved chunks with a cross-encoder model
- Query rewriting/expansion before retrieval

### 6. Generator
The LLM receives the retrieved chunks as context along with the user's question
and produces a grounded answer. A well-designed prompt instructs the model to
explicitly say when the context does not contain enough information, rather
than hallucinating an answer.

## Failure Modes
- **No relevant context**: if similarity scores for all retrieved chunks fall
  below a threshold, the system should return a fallback message instead of
  forcing an answer.
- **Context overflow**: retrieving too many chunks can exceed the model's
  context window or dilute attention on the most relevant passages.
- **Stale embeddings**: if source documents change, embeddings must be
  regenerated to keep the vector store in sync.
