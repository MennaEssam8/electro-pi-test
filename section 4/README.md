# Section 4 — Model Deployment

## Model & justification
`Qwen/Qwen2.5-0.5B-Instruct`, served on CPU, loaded in `bfloat16` (not
`float32`) specifically to keep the container's memory footprint low enough
to load reliably inside Docker Desktop's default memory limits on Windows —
an earlier `float32` version got OOM-killed by Docker mid-weight-load
(~2GB just for fp32 weights, before framework overhead). If you hit an
"unexpected EOF" error when running the container, increase Docker
Desktop's memory allocation (Settings -> Resources -> Memory, at least
4GB) in addition to using `bfloat16`.

Deliberately smaller than the 1.5B model used in Section 3 — this section is scored on deployment
engineering (serving architecture, containerization, streaming, latency
awareness), not model capability, and my local machine has no GPU. A smaller
model keeps CPU inference latency low enough that the 10-concurrent-request
load test produces meaningful, readable numbers instead of everything simply
being slow. The same `app.py` works unmodified with any Hugging Face
causal LM — swapping `MODEL_ID` is enough to point it at the Section 3 model
on GPU-enabled hardware.

## Architecture choice: FastAPI, not vLLM/TGI
I chose a hand-rolled FastAPI service over vLLM/TGI for this take-home
because: (1) it runs on CPU with no extra setup — vLLM's CPU backend is
possible but has a much heavier install footprint and less mature CPU
support than its GPU path; (2) it makes the streaming and request-handling
logic fully visible in ~100 lines, which is more useful for this assessment
than a production-grade serving engine's internals being hidden behind a
CLI flag. For an actual production GPU deployment I would switch to vLLM —
see the write-up below for why.

## Endpoints
- `GET /health` — liveness check
- `POST /generate` — full response in one JSON payload, includes `latency_sec`
- `POST /generate/stream` — token-by-token streaming via Server-Sent Events

## Running with Docker

```bash
cd section 4
docker build -t quant-demo-api .
docker run -p 8000:8000 quant-demo-api
```

First run downloads the model inside the container (~1GB), so the first
request will be slow while it's loading — wait for the "Model loaded." log
line before testing.

Test it:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is FastAPI?", "max_new_tokens": 60}'

curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is FastAPI?", "max_new_tokens": 60}'
```

## Load / latency test

With the container running, in a second terminal:
```bash
pip install httpx
python load_test.py
```

This fires 10 concurrent requests at both endpoints and reports:
- **Non-streaming**: total per-request latency (min/max/avg)
- **Streaming**: time-to-first-token and total latency (min/max/avg)

### Results

Measured locally via Docker (CPU-only, `bfloat16`), 10 concurrent requests,
30 max new tokens per request, prompt: "Explain what a REST API is in two sentences."

```
=== Non-streaming /generate: 10 concurrent requests ===
Wall-clock time for all 10 requests: 94.32s
Per-request total latency: min=92.53s max=94.32s avg=93.82s

=== Streaming /generate/stream: 10 concurrent requests ===
Wall-clock time for all 10 requests: 82.76s
Per-request total latency: min=79.60s max=82.76s avg=81.91s
Time-to-first-token: min=9.06s max=10.97s avg=10.40s
```

**Analysis:** all 10 requests took roughly the same total time (~93-94s for
non-streaming, ~80-83s for streaming) regardless of when they started —
`min` and `max` per-request latency are within ~2s of each other across all
10 requests. This is direct evidence of the sequential-processing behavior
described below: the requests aren't actually running in parallel on this
CPU-bound single-process server, they're effectively queued and processed
one at a time, so every request finishes at roughly the same wall-clock
moment regardless of arrival order. A single request's true generation time
for 30 tokens is roughly 94s / 10 ≈ 9-10s — which matches almost exactly
the ~9-11s time-to-first-token measured on the streaming endpoint, since
TTFT there is dominated by waiting for the requests ahead of it in the
queue, not by the model itself being slow to produce one token.

The streaming endpoint's total latency (~82s) being lower than the
non-streaming endpoint's (~94s) also makes sense: `TextIteratorStreamer`
starts yielding tokens as they're produced in the background thread, so the
client-side "done" timer stops as soon as the last token is read rather than
waiting for the full response to be assembled server-side and returned in
one shot.

## Write-up: Scaling to 50 concurrent users in production

The current setup — one FastAPI process, one model instance, CPU inference —
does not scale past a handful of concurrent users. My load test above proves
this directly: all 10 requests finished within ~2 seconds of each other
(min 92.5s, max 94.3s for non-streaming), regardless of which request
started first. That's the signature of **sequential processing**, not
parallelism — `model.generate()` holds the GIL-bound CPU compute for the
entire duration of each request, so 10 "concurrent" requests to this single
process are actually served one at a time internally. A single request
alone takes roughly 9-10s for 30 tokens; with 10 requests queued behind each
other, the last one waits ~94s. At 50 concurrent users this queueing would
compound into a multi-minute wait — completely unusable.

To actually support 50 concurrent users, I would change several things:

1. **Move to a GPU and a real inference server (vLLM or TGI).** This is the
   single biggest lever. vLLM implements **continuous batching**: instead of
   processing one request fully before starting the next, it batches
   multiple in-flight requests together at the token-generation level, so
   GPU compute is shared efficiently across concurrent users instead of
   serialized. This alone typically gives an order-of-magnitude throughput
   improvement over naive `model.generate()` serving.
2. **Add a request queue (e.g. Redis + a worker pool, or vLLM's built-in
   scheduler).** Even with batching, there's a limit to how many requests
   can be in-flight at once without degrading per-request latency. A queue
   lets me bound the batch size, apply backpressure (return a "server busy,
   retry" response) instead of unbounded memory growth, and prioritize
   requests if needed.
3. **Horizontal autoscaling.** Run multiple replicas of the inference
   service behind a load balancer (e.g. Kubernetes HPA scaling on GPU
   utilization or queue depth), so traffic spikes beyond one GPU's capacity
   spill over to additional instances rather than queueing indefinitely.
4. **Caching.** For any repeated or near-duplicate prompts (common in
   support/FAQ-style use cases), a semantic cache (embed the prompt, check
   for a similar cached response above a similarity threshold) can skip
   inference entirely for a meaningful fraction of traffic.
5. **Separate the streaming transport from generation.** At higher scale I'd
   move token streaming to a message broker (e.g. Redis pub/sub or
   WebSockets fed from the inference worker) rather than holding an HTTP
   connection open per request in the API process itself, so the API layer
   and the GPU inference layer can scale independently.

In short: the current implementation demonstrates the serving contract
(streaming, latency measurement, containerization) correctly, but the
execution engine underneath (`transformers.generate()` on CPU, one request
at a time) is the part that would be replaced wholesale — not tuned — to
reach real production concurrency.