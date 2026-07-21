"""
Reads results_fp16.json, results_bnb4bit.json, results_gguf.json (whichever
exist) and prints a comparison table.

Usage:
    python compare.py
"""

import json
import os
from tabulate import tabulate

FILES = {
    "fp16 (baseline)": "results_fp16.json",
    "bitsandbytes 4-bit (NF4)": "results_bnb4bit.json",
    "GGUF Q4_K_M": "results_gguf.json",
}


def main():
    rows = []
    for label, filename in FILES.items():
        if not os.path.exists(filename):
            print(f"Skipping '{label}' - {filename} not found (run the corresponding script first)")
            continue
        with open(filename) as f:
            data = json.load(f)

        size_field = data.get("gguf_file_size_mb", data.get("peak_memory_mb"))
        rows.append([
            label,
            f"{data['peak_memory_mb']:.0f} MB",
            f"{data['avg_tokens_per_sec']:.1f} tok/s",
        ])

    if not rows:
        print("No result files found. Run run_fp16.py, run_bnb4bit.py, and/or run_gguf.py first.")
        return

    print(tabulate(rows, headers=["Config", "Peak Memory", "Avg Throughput"], tablefmt="github"))
    print("\nOpen each results_*.json to compare generated text quality side by side "
          "for the same 5 prompts.")


if __name__ == "__main__":
    main()
