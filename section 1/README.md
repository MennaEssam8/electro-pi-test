# Section 1 — LiveKit Agents (Real-time Voice AI)

## Overview

This section implements a real-time voice support agent using the **LiveKit Agents SDK**. The agent acts as a customer support assistant for a fictional food delivery service (**QuickBite**) and demonstrates:

- Real-time **Speech-to-Text (STT)** using **Deepgram**
- **LLM reasoning and function calling** using **Groq (Llama 3.3 70B Versatile)**
- Real-time **Text-to-Speech (TTS)** using **Cartesia**
- Automatic tool invocation through `@function_tool`
- A mocked order database used as a stand-in for a real backend API

The only mocked component is the order lookup (`MOCK_ORDERS`), which replaces what would normally be an external order management service.

---

## Architecture

```text
          User Speech
               │
               ▼
      Deepgram Speech-to-Text
               │
               ▼
   Groq Llama 3.3 70B Versatile
               │
               ▼
        Function Calling
         ├─────────────────────┐
         │                     │
         ▼                     ▼
get_order_status()   get_delivery_estimate()
               │
               ▼
        Cartesia Text-to-Speech
               │
               ▼
         Spoken Response
```

---

## Technologies

- Python 3.10+
- LiveKit Agents SDK
- Deepgram STT
- Groq LLM
- Cartesia TTS
- python-dotenv

---

## Setup

Install the dependencies:

```bash
cd section 1
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

---

## Running

Start the agent:

```bash
python agent.py console
```

The application starts a local interactive voice session.

Speak naturally through your microphone.

### Example Conversation

```
User:
Hi, I'd like to check my order.

Assistant:
Sure! Could you tell me your order ID?

User:
12345

Assistant:
Your order is currently out for delivery and should arrive in about 12 minutes.
```

Invalid order example:

```
User:
99999

Assistant:
I'm sorry, I couldn't find an order with that ID.
Could you please check the number and try again?
```

---

## Function Tools

The assistant exposes two tools that the LLM can invoke automatically.

### `get_order_status(order_id: str)`

Returns the current status of an order.

Example:

```
Order 12345
→ Out for delivery
```

---

### `get_delivery_estimate(order_id: str)`

Returns the estimated remaining delivery time.

Example:

```
Order 12345
→ Approximately 12 minutes
```

The language model automatically decides when to invoke each function based on the user's request. Tool invocations are visible in the terminal logs, demonstrating that real function calling is taking place rather than the model answering from memory.

---

## Task 1.2 (Bonus)

The bonus task was implemented using real speech providers.

| Pipeline Component | Provider |
|--------------------|----------|
| Speech-to-Text | Deepgram |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Text-to-Speech | Cartesia |

Current `AgentSession` configuration:

```python
session = AgentSession(
    stt=deepgram.STT(),
    llm=groq.LLM(
        model="llama-3.3-70b-versatile"
    ),
    tts=cartesia.TTS(),
)
```

Because LiveKit separates the STT, LLM, and TTS components, replacing a provider only requires changing a single constructor.

Examples:

- Deepgram → OpenAI Whisper
- Cartesia → ElevenLabs
- Groq → OpenAI or Anthropic

No changes are required to the `SupportAgent` class or the function tools.

This modular architecture keeps the implementation vendor-independent.

---

---

# Write-up: Production Scaling (50 Concurrent Users)

The current implementation is designed as a simple single-container deployment suitable for demonstration purposes. If this service needed to support around **50 concurrent users** in production, I would improve the architecture in several ways.

### Request Queueing

Instead of letting every request compete directly for model inference, I would introduce a request queue (for example, Redis with Celery or another message broker). Queueing prevents the server from becoming overloaded during traffic spikes and provides more predictable response times.

### Dynamic Batching

If the inference backend supports it (such as vLLM or Text Generation Inference), I would enable dynamic batching. Multiple requests arriving within a short time window can be grouped into a single GPU inference batch, significantly improving throughput while only slightly increasing latency.

### Autoscaling

Rather than relying on a single container, I would deploy multiple API replicas behind a load balancer. Using Kubernetes Horizontal Pod Autoscaling (HPA) or a cloud autoscaling service, the number of replicas could automatically increase during periods of high traffic and decrease when demand falls, improving both availability and cost efficiency.

### Caching

Some requests are repeated frequently, such as health checks, common questions, or identical prompts. I would introduce Redis as a cache to store these responses and reduce unnecessary model inference. Caching would also be useful for storing user session information or conversation state.

### Optimized Model Serving

For larger workloads, I would replace the basic Transformers-based inference with a dedicated inference server such as **vLLM**, which provides continuous batching, efficient KV-cache management, and higher GPU utilization. This generally delivers much better throughput than running a standard Hugging Face model directly.

### Monitoring

Finally, I would add monitoring and observability using tools such as Prometheus and Grafana to track latency, throughput, GPU utilization, memory usage, request failures, and queue length. These metrics would help identify performance bottlenecks before they affect users.

Overall, these improvements would allow the service to scale from a simple demonstration into a production-ready system capable of serving dozens of concurrent users with lower latency, higher reliability, and better resource utilization.