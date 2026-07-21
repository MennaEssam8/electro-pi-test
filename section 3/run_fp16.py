"""
Baseline run: Qwen2.5-1.5B-Instruct at fp16, no quantization.

Usage:
    python run_fp16.py

Outputs:
    results_fp16.json  -- memory, throughput, and generated text per prompt
"""

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS, MAX_NEW_TOKENS
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} in fp16 on {device}...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    load_memory_mb = get_memory_mb()
    print(f"Model loaded. Memory after load: {load_memory_mb:.1f} MB")

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
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
        "config": "fp16",
        "model_id": MODEL_ID,
        "peak_memory_mb": peak_memory_mb,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "results": results,
    }

    with open("results_fp16.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (fp16) ===")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")
    print(f"Avg throughput: {summary['avg_tokens_per_sec']:.1f} tok/s")
    print("Saved to results_fp16.json")


if __name__ == "__main__":
    main()