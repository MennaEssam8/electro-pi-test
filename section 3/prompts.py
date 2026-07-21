"""
Shared, fixed set of 5 prompts used for ALL runs (fp16, bitsandbytes 4-bit,
GGUF) so that speed/quality comparisons are apples-to-apples.
"""

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "Write a Python function that returns the nth Fibonacci number using memoization.",
    "Summarize why retrieval-augmented generation helps reduce hallucination.",
    "A customer says their food delivery order never arrived. Write a short, empathetic response as a support agent.",
    "What are the trade-offs between using FAISS and Qdrant as a vector store?",
]

MAX_NEW_TOKENS = 200
