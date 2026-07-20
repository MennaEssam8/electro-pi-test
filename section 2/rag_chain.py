"""
RAG chain: retrieves relevant chunks for a question, answers with citations,
and explicitly refuses to answer when no relevant context is found.

Usage:
    python rag_chain.py "What is the advantage of Qdrant over FAISS?"

Environment:
        GROQ_API_KEY         -> uses llama-3.1-8b-instant via langchain-groq
"""

import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # reads GROQ_API_KEY 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Below this similarity score threshold, we consider retrieval to have
# failed and refuse to answer instead of letting the LLM hallucinate.
RELEVANCE_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = """You are a precise assistant that answers ONLY using the provided context.

Rules:
- If the context does not contain enough information to answer the question,
  respond exactly with: "I don't have enough information in the provided documents to answer that."
- Never use outside knowledge, even if you know the answer.
- Always cite which source file each part of your answer comes from, using the
  format [source: filename].

Context:
{context}

Question: {question}
"""


def get_llm():
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or GROQ_API_KEY in your environment."
    )


def format_docs_with_sources(docs):
    parts = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "unknown"))
        parts.append(f"[source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return vectorstore, chain


def answer_question(question: str):
    vectorstore, chain = build_chain()

    # Explicit "no relevant context" guardrail based on similarity score,
    # checked BEFORE calling the LLM at all.
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=3)

    print("Retrieved chunk scores:")
    for doc, score in results_with_scores:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  {score:.3f}  [{source}]")

    if not results_with_scores or all(score < RELEVANCE_SCORE_THRESHOLD for _, score in results_with_scores):
        return "I don't have enough information in the provided documents to answer that. (No chunk passed the relevance threshold.)"

    return chain.invoke(question)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag_chain.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"\nQ: {question}\n")
    print(answer_question(question))