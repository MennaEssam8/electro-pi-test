"""
Basic load/latency test: fires N concurrent requests at the running service
and reports time-to-first-token (streaming endpoint) and total latency
(non-streaming endpoint).

Usage:
    # make sure the server is running first (docker run ... or uvicorn app:app)
    python load_test.py
"""

import asyncio
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
N_CONCURRENT = 10
PROMPT = "Explain what a REST API is in two sentences."
MAX_NEW_TOKENS = 30  # kept small deliberately -- see README note on CPU contention
REQUEST_TIMEOUT = 300  # generous timeout so we measure the real (slow) latency instead of failing


async def call_generate(client: httpx.AsyncClient, request_id: int):
    start = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/generate",
        json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=REQUEST_TIMEOUT,
    )
    total_latency = time.perf_counter() - start
    resp.raise_for_status()
    print(f"  [non-stream] request {request_id} done in {total_latency:.1f}s")
    return {"request_id": request_id, "total_latency_sec": total_latency}


async def call_generate_stream(client: httpx.AsyncClient, request_id: int):
    start = time.perf_counter()
    ttft = None
    async with client.stream(
        "POST",
        f"{BASE_URL}/generate/stream",
        json={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        async for chunk in resp.aiter_bytes():
            if ttft is None and chunk:
                ttft = time.perf_counter() - start
    total_latency = time.perf_counter() - start
    print(f"  [stream] request {request_id} done in {total_latency:.1f}s (ttft={ttft:.1f}s)")
    return {"request_id": request_id, "ttft_sec": ttft, "total_latency_sec": total_latency}


async def run_load_test(concurrent_fn, label):
    print(f"\n=== {label}: {N_CONCURRENT} concurrent requests ===")
    async with httpx.AsyncClient() as client:
        overall_start = time.perf_counter()
        tasks = [concurrent_fn(client, i) for i in range(N_CONCURRENT)]
        results = await asyncio.gather(*tasks)
        overall_elapsed = time.perf_counter() - overall_start

    latencies = [r["total_latency_sec"] for r in results]
    print(f"Wall-clock time for all {N_CONCURRENT} requests: {overall_elapsed:.2f}s")
    print(f"Per-request total latency: min={min(latencies):.2f}s "
          f"max={max(latencies):.2f}s avg={sum(latencies)/len(latencies):.2f}s")

    if "ttft_sec" in results[0]:
        ttfts = [r["ttft_sec"] for r in results if r["ttft_sec"] is not None]
        if ttfts:
            print(f"Time-to-first-token: min={min(ttfts):.2f}s "
                  f"max={max(ttfts):.2f}s avg={sum(ttfts)/len(ttfts):.2f}s")

    return results


async def main():
    # Sanity check the server is up first
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print("Server health check OK:", resp.json())

    await run_load_test(call_generate, "Non-streaming /generate")
    await run_load_test(call_generate_stream, "Streaming /generate/stream")


if __name__ == "__main__":
    asyncio.run(main())