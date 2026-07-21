"""
FastAPI service wrapping Qwen2.5-0.5B-Instruct for text generation.

Two endpoints:
  POST /generate          -> full response in one JSON payload
  POST /generate/stream   -> token-by-token streaming (Server-Sent Events)

Chosen model: Qwen2.5-0.5B-Instruct, deliberately smaller than the model used
in Section 3's quantization comparison — this section is about deployment
engineering (serving, streaming, containerization, load testing), not model
capability, and a smaller model keeps CPU inference latency reasonable for
the load test in load_test.py. See README for the full justification.
"""

import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per-request -- avoids paying model load cost
    # on every call, which would dominate latency for a small model.
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    print("Model loaded.")
    yield
    model_state.clear()


app = FastAPI(title="Quantization Demo API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Non-streaming: returns the full response plus latency info."""
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

    messages = [{"role": "user", "content": req.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=req.max_new_tokens, do_sample=False
        )
    elapsed = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "response": text,
        "latency_sec": elapsed,
        "n_tokens": len(generated_ids),
    }


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """Streaming: yields tokens as they're generated (chunked text/event-stream)."""
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

    messages = [{"role": "user", "content": req.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )

    # generate() blocks until done, so it must run in a background thread
    # while we read from the streamer in the main request coroutine.
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def event_stream():
        for token_text in streamer:
            yield f"data: {token_text}\n\n"
        yield "data: [DONE]\n\n"
        thread.join()

    return StreamingResponse(event_stream(), media_type="text/event-stream")