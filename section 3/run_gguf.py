"""
Quantized run: Qwen2.5-1.5B-Instruct as a GGUF Q4_K_M build, via llama-cpp-python.

This downloads a pre-quantized GGUF file from the official Qwen GGUF repo on
Hugging Face (no manual conversion needed) and runs it with llama.cpp's
Python bindings.

Usage:
    python run_gguf.py

Outputs:
    results_gguf.json  -- memory, throughput, and generated text per prompt
"""

import json
import time
import os
import psutil
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from prompts import PROMPTS, MAX_NEW_TOKENS

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"  # ~1GB, 4-bit K-quant


def get_memory_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def main():
    print(f"Downloading {FILENAME} from {REPO_ID} (cached after first run)...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    print(f"Loading GGUF model from {model_path}...")
    # n_gpu_layers=-1 offloads all layers to GPU if llama-cpp-python was built
    # with CUDA support; falls back to CPU automatically otherwise.
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    load_memory_mb = get_memory_mb()
    gguf_file_size_mb = os.path.getsize(model_path) / (1024 ** 2)
    print(f"Loaded. Process RSS after load: {load_memory_mb:.1f} MB "
          f"(GGUF file on disk: {gguf_file_size_mb:.1f} MB)")

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]

        start = time.perf_counter()
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start

        generated_text = output["choices"][0]["message"]["content"]
        n_tokens = output["usage"]["completion_tokens"]
        tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0

        total_tokens += n_tokens
        total_time += elapsed

        print(f"\n[{i+1}/{len(PROMPTS)}] {elapsed:.2f}s | {tokens_per_sec:.1f} tok/s")
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text[:200]}...")

        results.append({
            "prompt": prompt,
            "output": generated_text,
            "elapsed_sec": elapsed,
            "n_tokens": n_tokens,
            "tokens_per_sec": tokens_per_sec,
        })

    peak_memory_mb = get_memory_mb()

    summary = {
        "config": "gguf-q4_k_m",
        "model_id": f"{REPO_ID}/{FILENAME}",
        "peak_memory_mb": peak_memory_mb,
        "gguf_file_size_mb": gguf_file_size_mb,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "results": results,
    }

    with open("results_gguf.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (GGUF Q4_K_M) ===")
    print(f"Process RSS: {peak_memory_mb:.1f} MB | File size on disk: {gguf_file_size_mb:.1f} MB")
    print(f"Avg throughput: {summary['avg_tokens_per_sec']:.1f} tok/s")
    print("Saved to results_gguf.json")


if __name__ == "__main__":
    main()
