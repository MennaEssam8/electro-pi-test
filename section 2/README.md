# Section 2 — LangChain RAG Pipeline

## Document Set
I used my own domain documents (as permitted by the task) instead of the
provided files, covering topics from my own AI engineering work:
- `docs/rag_architecture.md` — RAG pipeline components and failure modes
- `docs/vector_store_comparison.md` — FAISS vs Chroma vs Qdrant
- `docs/responsible_ai_notes.md` — human-in-the-loop and bias auditing patterns

## Design
- **Chunking**: `RecursiveCharacterTextSplitter`, 600 chars / 100 overlap,
  splitting on markdown headers first so related content stays together.
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` — no API key
  or cost for the embedding step, runs on CPU.
- **Vector store**: Chroma, persisted locally to `./chroma_db`.
- **LLM**: pluggable — uses`GROQ_API_KEY` (Llama 3.1 8B). This mirrors how I designed the LLM router in my multi-agent NLP project.
- **Citations**: each retrieved chunk is tagged with its source filename and
  passed to the LLM, which is instructed to cite `[source: filename]` inline.
- **No-hallucination guardrail**: before calling the LLM at all, I run
  `similarity_search_with_relevance_scores` and check the top score against
  `RELEVANCE_SCORE_THRESHOLD = 0.3`. If nothing clears the bar, the pipeline
  returns a fixed "not enough information" message and never calls the LLM —
  this guarantees no hallucination on out-of-scope questions, rather than
  just hoping the prompt instruction is followed.

## Setup

```bash
cd section 2
pip install -r requirements.txt

# one-time: build the vector store
python ingest.py

# GROQ_API_KEY=your_key_here

python rag_chain.py "What is the advantage of Qdrant over FAISS?"
```

## Example Questions & Answers

Real output from `rag_chain.py`, run after `ingest.py` (relevance threshold = 0.3).

---

**Q1: Explain the RAG architecture.**

```
Retrieved chunk scores:
  0.436  [rag_architecture.md]
  0.168  [vector_store_comparison.md]
  0.121  [rag_architecture.md]
```

> A Retrieval-Augmented Generation (RAG) system combines a retriever and a generator to answer questions grounded in a document collection, instead of relying purely on the LLM's parametric knowledge. [source: rag_architecture.md]
>
> The core components of the RAG architecture include:
> 1. Document Loader: Responsible for reading raw files (PDF, Markdown, HTML) and converting them into a unified `Document` object with `page_content` and `metadata` (source, page number)
> 2. Retriever: Given a query, embeds it and returns the top-k most similar chunks. Retrieval quality can be improved with hybrid search, re-ranking with a cross-encoder, or query rewriting/expansion. [source: rag_architecture.md]
> 3. Generator: The LLM receives the retrieved chunks as context along with the user's question and produces a grounded answer, instructed to say when context is insufficient rather than hallucinate. [source: rag_architecture.md]

At the default `k=3`, this answer skipped the Text Splitter, Embedding Model,
and Vector Store components (their content simply wasn't among the top-3
retrieved chunks). Re-running the same question with `k=5` recovered all six
components — see the write-up below.

---

**Q2: What is the advantage of Qdrant over FAISS?**

```
Retrieved chunk scores:
  0.459  [vector_store_comparison.md]
  0.421  [vector_store_comparison.md]
```

> Qdrant's advantages over FAISS include:
> - Multi-collection support: separate collections can be created for different embedding types [source: vector_store_comparison.md]
> - Payload filtering: structured metadata can be attached to each point and filtered on at query time [source: vector_store_comparison.md]
> - Persistence and horizontal scaling: Qdrant supports on-disk storage [source: vector_store_comparison.md]
> - Ability to query multiple collections independently and merge results [source: vector_store_comparison.md]

---

**Q3: Explain Human-in-the-loop Enforcement.**

```
Retrieved chunk scores:
  0.458  [responsible_ai_notes.md]
  0.247  [responsible_ai_notes.md]
  0.205  [responsible_ai_notes.md]
```

> For any system that produces decisions affecting people (e.g. hiring, moderation, approvals), the final decision field should be structurally unable to be set automatically by the AI pipeline. One pattern is to leave the decision field NULL in the data model until a human reviewer explicitly calls a dedicated decision endpoint. This makes human review an architectural guarantee rather than a policy that can be silently skipped. [source: responsible_ai_notes.md]

---

**Q4: What is the capital of Egypt?** *(deliberately out-of-scope)*

```
Retrieved chunk scores:
  0.080  [vector_store_comparison.md]
  0.040  [vector_store_comparison.md]
  0.005  [rag_architecture.md]
```

> I don't have enough information in the provided documents to answer that. (No chunk passed the relevance threshold.)

This confirms the no-hallucination guardrail works as designed: even though
some chunks were retrieved, all three scores fell well below the 0.3
threshold, so the pipeline refused to answer rather than calling the LLM at
all — no API call was made for this question.

## Write-up: Improving Retrieval on Longer Documents

If answer quality degraded on longer documents, I would make these changes,
roughly in order of effort vs. payoff:

1. **Tune retrieval parameters (k) first — cheapest fix.** I tested this
   directly: with the "Explain the RAG architecture" question, `k=3`
   retrieved chunks covering only 3 of the 6 components described in
   `rag_architecture.md` (Document Loader, Retriever, Generator), missing
   Text Splitter, Embedding Model, and Vector Store. Re-running the same
   question with `k=5` recovered all six components in the answer. For
   broad questions spanning a document with many sub-sections, a higher `k`
   clearly improved completeness — the trade-off is more tokens in the
   context window and slightly higher risk of including a
   marginally-relevant chunk that dilutes the answer.
2. **Hybrid search.** Pure dense retrieval misses exact keyword matches
   (IDs, acronyms, proper nouns). Combining BM25 sparse search with dense
   vector search via a weighted ensemble retriever usually gives the biggest
   quality jump for the least engineering effort, especially for longer
   documents where more of the query terms need exact matching.
3. **Re-ranking.** Retrieve a larger candidate set (top 15–20) with the fast
   vector search, then re-rank with a cross-encoder (e.g.
   `cross-encoder/ms-marco-MiniLM-L-6-v2`) to pick the final top 3–5. Cross-
   encoders score the query and chunk jointly, which is far more accurate
   than cosine similarity on independently-embedded vectors, at the cost of
   extra latency.
4. **Smaller, structure-aware chunks.** For long documents, I'd switch from
   fixed character chunking to splitting on document structure (headers,
   sections) with smaller chunk sizes (~300–400 chars) so each chunk is more
   topically focused, and attach the parent section title as metadata so it
   can still be surfaced for context even if not embedded directly.
5. **Query rewriting/expansion.** For long technical documents, user
   questions are often phrased differently than the document's terminology.
   An LLM-based query rewrite step before retrieval (or generating multiple
   paraphrased queries and merging results) can recover relevant chunks that
   a single embedding of the raw question would miss.
