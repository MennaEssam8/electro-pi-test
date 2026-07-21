# Section 3 — Quantization

## Model
`Qwen/Qwen2.5-1.5B-Instruct` — small enough to run full fp16 and multiple
quantized variants on a single consumer GPU, while still being a real
instruction-tuned chat model (not a toy).

## What's compared
| Variant | Technique | Script |
|---|---|---|
| fp16 baseline | none | `run_fp16.py` |
| 4-bit NF4 | `bitsandbytes` via `transformers` | `run_bnb4bit.py` |
| Q4_K_M GGUF | `llama.cpp` (via `llama-cpp-python`) | `run_gguf.py` *(optional — see Known Limitations)* |

All runs use the exact same 5 fixed prompts from `prompts.py`, generating up
to 200 tokens per prompt with greedy decoding (temperature 0), so the
comparison is apples-to-apples.

## Running (recommended: Google Colab)

My local machine has no GPU and limited disk space, so I ran this section on
Google Colab's free T4 GPU instead of locally.

1. Upload `quantization_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. `Runtime -> Change runtime type -> T4 GPU`
3. Run all cells top to bottom
4. The last cell downloads `results_fp16.json` and `results_bnb4bit.json` —
   save both into this folder
5. Run `python compare.py` locally (just needs `tabulate`, no GPU) to
   regenerate the summary table, or read the numbers directly from the
   notebook output

The standalone scripts (`run_fp16.py`, `run_bnb4bit.py`, `run_gguf.py`) are
still included and work identically on any CUDA-enabled machine — the
notebook just avoids the local environment issues (no GPU, low disk space,
Windows build issues with `llama-cpp-python`) documented below.

## Known Limitations

I did not run the GGUF variant. My local disk (C: drive) didn't have enough
free space for an additional ~1GB model download, `llama-cpp-python` hit a
Windows long-path build issue during install, and my machine has no GPU at
all (which is also why I moved fp16/bitsandbytes to Colab). The GGUF script
(`run_gguf.py`) is included and should work as-is on a machine with more
disk headroom and a working `llama-cpp-python` install; the write-up below
reasons about GGUF's trade-offs from architecture/tooling knowledge rather
than from a measurement I collected myself, and I've flagged it as such.

## Setup (if running scripts directly on a CUDA machine instead of Colab)

```bash
cd section 3
pip install -r requirements.txt
```

## Results

Measured on Google Colab (T4 GPU, free tier).

| Precision | Model size | Peak memory (load) | Avg throughput | Quality (5 fixed prompts) |
|---|---|---|---|---|
| fp16 | ~3.1 GB | 2966 MB | 23.2 tok/s | Baseline reference |
| bitsandbytes 4-bit NF4 | ~1.1 GB (in-memory) | 1166 MB | 8.6 tok/s | Equivalent to fp16 on all 5 prompts |
| GGUF Q4_K_M | ~1.0 GB (on disk) | not measured — see Known Limitations | not measured | not measured |

**Memory**: 4-bit cut peak memory by ~61% (2966 MB -> 1166 MB), as expected —
this is the whole point of the technique.

**Speed**: counter-intuitively, 4-bit was ~2.7x *slower* than fp16 (8.6 vs
23.2 tok/s), not faster. I initially expected quantization to speed things up
too, so I double-checked this wasn't a fluke — the pattern held consistently
across all 5 prompts (e.g. prompt 2: 7.95s at fp16 vs 34.89s at 4-bit). The
explanation is in the write-up below: bitsandbytes trades memory for speed,
it does not improve both.

**Quality**: I read through all 5 outputs in `results_fp16.json` and
`results_bnb4bit.json` side by side. Both versions gave correct, coherent,
on-topic answers to all 5 prompts — same core content (e.g. both correctly
described memoization for the Fibonacci prompt, both correctly identified
the FAISS vs Qdrant trade-offs). Differences were purely stylistic: the
4-bit version tended to add more markdown structure (numbered
sub-headings) on the longer prompts, while fp16 stayed in plain paragraphs.
At this model size (1.5B) and bit-width (4-bit NF4), I did not observe any
quality degradation — no repetition, no factual errors, no incoherent
output in either version.

## Write-up: GPTQ/AWQ vs bitsandbytes vs GGUF for production

*(Note: the bitsandbytes vs fp16 comparison below is grounded in the numbers
I measured on Colab above, including the counter-intuitive speed result. The
GGUF discussion is based on how the tooling and inference engine work, since
I wasn't able to run it empirically this time — see Known Limitations.)*

**bitsandbytes** is the easiest to reach for because it plugs directly into
`transformers` with a couple of config lines and requires no separate
calibration step — you load the fp16 weights and quantize on the fly. That
convenience is also its weakness in production: quantization happens at load
time, adds startup latency, and — as I measured directly — NF4 in
bitsandbytes was **~2.7x slower** than fp16 on my test (8.6 vs 23.2 tok/s),
despite using 61% less memory. This isn't a fluke: bitsandbytes still
dequantizes weights back to fp16 on the fly for every matmul, so you pay a
dequantization cost on top of the matmul itself, and there's no fused
low-bit CUDA kernel doing the compute directly in 4-bit the way GPTQ/AWQ
have. In other words, bitsandbytes optimizes for memory, not speed — the two
are not the same trade-off, and I'd have assumed they moved together before
running this myself. I'd reach for bitsandbytes when memory is the
bottleneck (e.g. fitting a model that otherwise wouldn't fit on the
available GPU) or during development/experimentation, but not when inference
throughput matters and memory isn't actually constrained.

**GPTQ/AWQ** are calibration-based: they use a small calibration dataset to
decide how to quantize weights layer-by-layer, which generally preserves
quality better than bitsandbytes at the same bit-width, and inference is
faster because there's dedicated CUDA kernels for the pre-quantized format
rather than on-the-fly dequantization. The cost is an upfront quantization
step (minutes to hours depending on model size) that has to be redone if you
fine-tune or swap the base model. I'd pick GPTQ/AWQ for a production
**GPU-served** endpoint where you quantize once and serve millions of
requests — the one-time calibration cost amortizes quickly, and the
inference-speed edge over bitsandbytes matters at scale.

**GGUF (llama.cpp)** is the right choice when the deployment target is CPU,
Apple Silicon, or resource-constrained edge hardware rather than a
datacenter GPU. It's not primarily a "better quantization algorithm" pitch —
its real advantage is the inference engine: llama.cpp is extremely well
optimized for CPU inference and memory-mapped model loading, which GPTQ/AWQ
and bitsandbytes (both GPU/CUDA-first) don't target. I'd pick GGUF for
on-device or self-hosted-without-GPU scenarios, or when I need a single
portable model file that's easy to distribute and run with minimal
dependencies.

**In short**: bitsandbytes for fast iteration, GPTQ/AWQ for a GPU-served
production endpoint at scale, GGUF when the target hardware doesn't have a
capable GPU at all.