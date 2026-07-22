# Electro Pi – AI Engineer Technical Test

This repository contains my submission for the **Electro Pi AI Engineer Technical Assessment**. The project is organized into four independent sections, each demonstrating one of the required skills listed in the assessment.

---

# Project Structure

```text
electro-pi-test/
│
├── section 1/
│   ├── agent.py
│   ├── README.md
│   └── ...
│
├── section 2/
│   ├── ingest.py
│   ├── rag_chain.py
│   ├── README.md
│   └── ...
│
├── section 3/
│   ├── run_fp16.py
│   ├── run_4bit.py
│   ├── benchmark.py
│   ├── README.md
│   └── ...
│
├── section 4/
│   ├── app.py
│   ├── Dockerfile
│   ├── README.md
│   └── ...
│
├── requirements.txt
└── README.md
```

---

# Requirements

Before running the project, make sure you have:

- Python 3.10+
- Git
- Docker Desktop (required for Section 4)
- Internet connection (for downloading models and using cloud APIs)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/MennaEssam8/electro-pi-test
cd electro-pi-test
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
HF_TOKEN=your_huggingface_token
```

---

# Section 1 — LiveKit Agents

This section demonstrates:

- LiveKit Agents SDK
- Deepgram Speech-to-Text
- Groq LLM
- Cartesia Text-to-Speech
- Function Calling

Run:

```bash
cd "section 1"
python agent.py console
```

See **section 1/README.md** for implementation details.

---

# Section 2 — LangChain RAG

This section demonstrates:

- Document ingestion
- Chunking
- Embeddings
- FAISS vector database
- Retrieval-Augmented Generation
- Citation-based answers

### Build the vector database

```bash
cd "section 2"
python ingest.py
```

### Ask a question

```bash
python rag_chain.py "Explain the RAG architecture."
```

See **section 2/README.md** for implementation details.

---

# Section 3 — Quantization

This section demonstrates:

- FP16 baseline inference
- 4-bit quantization
- Throughput benchmarking
- Memory usage comparison
- Quality comparison

Run the FP16 model:

```bash
cd "section 3"
python run_fp16.py
```

Run the quantized model:

```bash
python run_4bit.py
```

Generate the benchmark:

```bash
python benchmark.py
```

See **section 3/README.md** for implementation details.

---

# Section 4 — Model Deployment

This section demonstrates:

- FastAPI inference server
- Docker deployment
- Streaming responses
- Load and latency testing

Build the Docker image:

```bash
cd "section 4"
docker build -t quant-demo-api .
```

Run the container:

```bash
docker run -p 8000:8000 quant-demo-api
```

Open the API documentation:

```
http://localhost:8000/docs
```

See **section 4/README.md** for implementation details.

---

# Technologies Used

- Python
- LiveKit Agents
- Deepgram
- Groq
- Cartesia
- LangChain
- FAISS
- Hugging Face Transformers
- BitsAndBytes
- FastAPI
- Docker
- PyTorch

---

# Assumptions

- API keys are provided through the `.env` file.
- Hugging Face models are downloaded automatically during the first run.
- Some sections require an internet connection to download models or access cloud APIs.
- Docker Desktop is required only for Section 4.

---

# Notes

Each section is self-contained and includes its own **README.md** explaining:

- Setup instructions
- How to run the code
- Design decisions
- Assumptions
- Required write-up for the corresponding task

This organization allows each section to be evaluated independently while sharing a common project structure.