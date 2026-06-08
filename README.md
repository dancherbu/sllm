# sLLM — Shallow Large Language Model

> Train a language model on your laptop CPU. No GPU. No cloud. No calculus. Just counting.

---

## What is sLLM?

sLLM is a **radical departure from traditional language models**. Instead of using gradient descent and backpropagation (calculus-heavy neural networks), it learns through **Hebbian associative counting** — the principle that "neurons that fire together, wire together."

The result is a language model that:

- **Trains on any CPU** — no GPU, no CUDA, no cloud compute
- **Uses < 2.5 GB RAM** during training
- **Produces a ~50 MB model file** (`brain.sllm`) that runs anywhere
- **Runs inference with near-zero power draw** — pure table lookups, no matrix multiplies
- **Learns continuously** — no separate train/inference modes
- **Is fully portable** — copy one file to any device and run it

## Architecture

sLLM consists of four components:

| Component | Binary | Role |
|-----------|--------|------|
| **sllm-core** | Library | Shared: tokenizer, brain, file format, RAG |
| **sllm-train** | `sllm-train` | Training engine (read+write to model) |
| **sllm-run** | `sllm-run` | Inference runner (read-only, HTTP API) |
| **sllm-cli** | `sllm` | Unified CLI tool |

The training engine and inference runner are **separate binaries** by design. The runner loads the model file via memory-mapped I/O in read-only mode — it never modifies the model.

## Quick Start

```bash
# Build everything
cargo build --release

# Train on English text first
sllm train --data ./data/tinystories/ --output ./models/base.sllm --phase english

# Train on code
sllm train --data ./data/python-code/ --output ./models/code.sllm --phase code --base ./models/base.sllm

# Start the inference server
sllm serve --model ./models/code.sllm

# Generate text
curl -X POST http://localhost:11435/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "code", "prompt": "def fibonacci(n):\n    ", "max_tokens": 64}'

# Interactive chat
sllm run code
```

## How It Works

### Learning (Zero Calculus)

Traditional LLMs adjust millions of floating-point weights by computing gradients through chain rule calculus. sLLM does something far simpler:

```
For every sequence of tokens seen during training:
    Count how often token B follows token A → increment integer counter
    Count how often token C follows [A, B] → increment integer counter
    Count how often token D follows [A, B, C] → increment integer counter
```

At inference, given a context of recent tokens, sLLM looks up the most frequent next tokens across multiple n-gram orders and samples from the resulting distribution.

### The Brain File (`brain.sllm`)

A self-contained binary file (~50 MB) containing:
- BPE vocabulary and merge rules
- Sparse n-gram count tables (2-gram through 5-gram)
- Optional RAG index data
- Model metadata (name, version, training stats)

Copy this one file to any device → load it → run it. No config files, no external dependencies.

## Project Philosophy

1. **No calculus** — Integer counting only. No gradients, no loss functions.
2. **CPU-only** — Runs on any hardware with a CPU. SIMD welcome, GPU forbidden.
3. **Edge-native** — Must fit on phones, tablets, Raspberry Pi.
4. **Portable brain** — One file, any device.
5. **Minimal dependencies** — Prefer hand-rolled over heavy frameworks.
6. **Rust** — Safety, speed, and `memmap2`.

## Target Languages

sLLM is trained to understand and generate:
- **Python**
- **JavaScript / TypeScript**
- **Rust**

## Building

```bash
# Requirements: Rust 1.75+ (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone <repo-url> sllm
cd sllm
cargo build --release

# Run tests
cargo test --workspace

# Check for warnings
cargo clippy --workspace
```

## Documentation

- [AGENT.md](./AGENT.md) — Full technical briefing for AI agents
- [FEATURES.md](./FEATURES.md) — Complete feature roadmap
- [sllm-core/](./sllm-core/) — Core library documentation
- [sllm-train/](./sllm-train/) — Training engine documentation
- [sllm-run/](./sllm-run/) — Inference runner documentation

## License

TBD

## Contributing

Read [AGENT.md](./AGENT.md) first. It contains all architectural constraints and coding standards.
