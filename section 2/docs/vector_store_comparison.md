# Vector Store Comparison: FAISS vs Qdrant vs Chroma

## FAISS
FAISS (Facebook AI Similarity Search) is a library, not a database. It runs
in-memory and offers extremely fast approximate nearest neighbor search over
large vector sets. It has no built-in metadata filtering, no persistence layer
out of the box, and no support for multiple isolated collections in a single
index. Best suited for single-purpose, read-heavy workloads where the index
is rebuilt periodically offline.

## Chroma
Chroma is a lightweight, developer-friendly vector database designed for rapid
prototyping. It supports persistence to disk, basic metadata filtering, and a
simple Python API that integrates directly with LangChain. It is a good default
choice for small to medium projects and take-home assessments where setup time
matters more than large-scale production guarantees.

## Qdrant
Qdrant is a production-grade vector database written in Rust. Its key advantages
over FAISS are:
- **Multi-collection support**: separate collections can be created for
  different embedding types (e.g. one collection for text chunks, another for
  image embeddings), each with its own distance metric and vector dimension.
- **Payload filtering**: structured metadata can be attached to each point and
  filtered on at query time (e.g. filter by document date or source type)
  without a separate database.
- **Persistence and horizontal scaling**: Qdrant supports on-disk storage,
  snapshots, and clustering for production deployments.

For systems that need to combine multiple modalities — for example, retrieving
both text passages and image captions relevant to a query — Qdrant's ability to
query multiple collections independently and merge results is a significant
architectural advantage over FAISS, which was not designed with multi-collection
workflows in mind.

## Recommendation
- Prototyping / take-home tests → Chroma
- Large offline batch search, no metadata needs → FAISS
- Production system with multiple embedding types or filtering needs → Qdrant
