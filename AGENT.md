# AGENT.md — sLLM Project Briefing

> **Read this file first.** It tells you everything you need to know to work on this project.

---

## What Is This?

**sLLM** (shallow Large Language Model) is a **gradient-free, CPU-only language model** built entirely in Rust. It does NOT use backpropagation, gradient descent, or any calculus-based optimization. Instead, it learns through **Hebbian associative counting** — "cells that fire together, wire together."

The model is designed for **edge deployment**: train on a laptop CPU, run on a phone/tablet/laptop with near-zero power draw. The trained model is a single portable file (`brain.sllm`, ~50 MB) loaded via memory-mapped I/O.

**This is NOT a small Transformer.** It is a fundamentally different architecture.

---

## Core Philosophy — Non-Negotiable Principles

1. **ZERO CALCULUS** — No gradients, no backprop, no optimizer states, no loss functions. Learning is done via integer count increments on conditional probability tables. If you find yourself importing a tensor library or computing derivatives, you are going the wrong direction.

2. **CPU-ONLY** — No CUDA, no GPU, no OpenCL. Everything must run on a standard CPU. SIMD intrinsics are welcome. GPU compute is not.

3. **EDGE-NATIVE** — The model must fit in ~50 MB, train under 2.5 GB RAM, and run inference under 200 MB RAM. It must be deployable to phones, tablets, laptops, and Raspberry Pi-class devices.

4. **PORTABLE BRAIN** — The `brain.sllm` file is a self-contained, platform-independent artifact. Copy it to any device → load it → run it. No external dependencies, no model shards, no config files needed.

5. **DUAL BINARY ARCHITECTURE** — The Training Engine (read+write) and Inference Runner (read-only) are separate binaries compiled from separate crates. The runner NEVER modifies the model file.

6. **RUST-FIRST** — All core logic is Rust. No Python wrappers, no C FFI for core paths. External tools (data scrapers, benchmarks) may use other languages.

7. **MINIMAL DEPENDENCIES** — Prefer hand-rolled implementations over heavy frameworks. Every dependency must justify its inclusion.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    sLLM System Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  sllm-core (library)                                        │
│  ├── tokenizer/   — BPE tokenizer (20k-24k vocab, multilingual)│
│  ├── brain/       — Associative count tables + Count-Min     │
│  ├── format/      — brain.sllm binary file format            │
│  └── rag/         — BM25 + SQLite retrieval index            │
│                                                              │
│  sllm-train (binary) — Read+Write training engine            │
│  ├── Streaming data pipeline (line-by-line, never full load) │
│  ├── Phased curriculum (Twi → English → Code → Personal)     │
│  └── Consolidation / "sleep" pass (prune + compact)          │
│                                                              │
│  sllm-run (binary) — Read-only inference runner              │
│  ├── mmap model loader (zero-copy)                           │
│  ├── HTTP API server (axum, port 11435)                      │
│  ├── Multi-model registry                                    │
│  └── RAG-augmented generation                                │
│                                                              │
│  sllm-cli (binary) — Unified CLI (like `ollama`)             │
│  └── sllm run | train | list | merge | serve | info          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## How the Brain Learns

```
Input text: "def fibonacci(n):"

Tokenizer: [def] [_fib] [onacci] [(] [n] [)] [:]

For each sliding window of tokens:
  - 2-gram: Count([def] → [_fib])++
  - 3-gram: Count([def, _fib] → [onacci])++
  - 4-gram: Count([def, _fib, onacci] → [(])++
  - 5-gram: Count([def, _fib, onacci, (] → [n])++

At inference:
  Given context [def, _fib, onacci, (, n, ), :],
  look up the highest-count next tokens across all n-gram orders,
  apply interpolation weighting (higher order = more weight),
  sample from the resulting distribution.
```

## The brain.sllm File Format

```
┌──────────────────────────────────────┐
│ Magic: "SLLM" (4 bytes)             │
│ Version: u16                         │
│ Header size: u32                     │
│ Model name (length-prefixed string)  │
│ Vocab size: u32                      │
│ Context window: u32                  │
│ N-gram orders: u8                    │
│ Total associations: u64              │
│ Training tokens seen: u64            │
│ Created timestamp: u64               │
│ Checksum: u32 (CRC32)               │
│ Section offsets (vocab, counts, rag) │
├──────────────────────────────────────┤
│ VOCAB SECTION                        │
│ BPE merges + token → string mappings │
├──────────────────────────────────────┤
│ COUNTS SECTION                       │
│ Sparse n-gram count tables           │
│ (sorted by context hash for mmap)    │
├──────────────────────────────────────┤
│ RAG INDEX SECTION (optional)         │
│ Embedded BM25 index data             │
└──────────────────────────────────────┘
```

---

## Directory Layout

```
sllm/
├── AGENT.md              ← YOU ARE HERE
├── DECISIONS.md          — All design decisions from each session
├── README.md             — Project overview and quickstart
├── FEATURES.md           — Feature roadmap with current/planned/future
├── Cargo.toml            — Workspace manifest
├── models/               — Default model storage
├── data/                 — Training data (gitignored)
│   ├── twi/              — Ashanti Twi text datasets
│   ├── english/          — TinyStories + Simple Wikipedia
│   ├── code/             — The Stack Processed V2 subsets
│   ├── personal/         — Symlinks to ~/Projects/
│   └── bootstrap/        — GPT-2 vocab, n-gram frequencies
├── scripts/              — Python data acquisition & monitoring
│   ├── download_twi.py   — Download 9 Twi datasets from HuggingFace
│   ├── download_english.py — TinyStories + Wikipedia
│   ├── download_code.py  — The Stack code subsets
│   ├── prepare_personal.py — Extract personal code from ~/Projects/
│   └── monitor.py        — Training & development dashboard
├── sllm-core/            — Shared library
├── sllm-train/           — Training engine binary
├── sllm-run/             — Inference runner binary
└── sllm-cli/             — CLI tool binary
```

---

## Current Status

> **Update this section as you work.** Mark phases complete as they are finished.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Foundation | ✅ COMPLETE | Cargo workspace, file format, tokenizer, count tables, runner, CLI |
| Phase 2: Training Pipeline | 🔵 IN PROGRESS | Data acquisition, streaming pipeline, curriculum |
| Phase 3: Inference Runner | ⚪ NOT STARTED | SSE streaming, stop sequences, model unload |
| Phase 4: RAG + CLI | ⚪ NOT STARTED | BM25 index, retrieval, unified CLI |
| Phase 5: HDC Projections | ⚪ NOT STARTED | Hyperdimensional Computing upgrade |
| Phase 6: Contrastive Learning | ⚪ NOT STARTED | Forward-Forward style learning |
| Phase 7: Agentic Loop | ⚪ NOT STARTED | Tool use, plan-code-test trajectories |
| Phase 8: Modular Experts | ⚪ NOT STARTED | Specialized sub-models + routing |

---

## Key Constraints for Agents

### DO:
- Write idiomatic Rust with proper error handling (`Result<T, E>`, `thiserror`)
- Use `#[cfg(test)]` for unit tests in every module
- Keep memory allocations minimal — prefer stack, slices, and arenas
- Use `memmap2` for file I/O in the runner
- Write doc comments on all public APIs
- Keep the `brain.sllm` format backwards-compatible (additive changes only)
- Use streaming/iterators for data processing (never load full datasets)

### DO NOT:
- Import PyTorch, TensorFlow, tch-rs, candle, burn, or any neural network framework
- Use floating-point weights or gradient computation
- Require GPU or CUDA in any code path
- Add dependencies without clear justification
- Break the train/infer binary separation
- Modify model files from the runner binary
- Use `unsafe` without a safety comment explaining why

### Testing:
- Run `cargo test --workspace` before committing
- Run `cargo clippy --workspace` — zero warnings policy
- Run `cargo fmt --check` — consistent formatting

---

## Development Machine

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 9955HX (Zen 5) — 16C/32T @ 4.56GHz |
| RAM | 64 GB DDR5 |
| Storage | 1TB NVMe SSD (510GB free) |
| GPU | RTX 3090 24GB — **intentionally unused** (no CUDA, no gradients) |
| OS | Linux x86_64 |
| Rust | 1.96.0+ |

---

## Training Data Strategy

| Phase | Dataset | Status | Purpose |
|-------|---------|--------|---------|
| 0. Ashanti Twi | 9 datasets from ghananlpcommunity (HF) | ✅ Downloaded | First language — Akan Twi patterns |
| 1. English | TinyStories (2.1M stories) + Simple Wikipedia (509k lines) | ✅ Downloaded | English fluency foundation — **expand over time** |
| 2. Code (public) | The Stack Processed V2 (Python, TS, JS, Rust) | 🔵 Downloading | Generic code patterns |
| 3. Code (personal) | ~/Projects/ (20,524 files, 4.5M lines) | ✅ Extracted | YOUR naming conventions, YOUR patterns |
| 4. Agentic | Synthetic trajectories (think→plan→code→test) | ⚪ Future | Multi-step reasoning patterns |

### English Fluency — Long-Term Plan

The sLLM must learn **full English fluency over time**. Current corpus is a starting point:
1. ✅ Phase 1: TinyStories + Wikipedia (basic patterns)
2. Future: Project Gutenberg (literature), OpenWebText (web), DailyDialog (conversation)
3. Future: Technical docs (man pages, READMEs, API docs)
4. Continuous incremental training — add more data without restarting

---

## Key Technical Decisions

1. **Tokenizer**: BPE with 20k–24k tokens, **multilingual** (English + Ashanti Twi + code-aware)
2. **Context window**: 128 tokens (sliding window for n-gram lookups)
3. **N-gram orders**: 2 through 5 simultaneously, interpolation-weighted
4. **Count storage**: HashMap during training → sorted arrays in .sllm file
5. **Memory efficiency**: Count-Min Sketch available but skipped during training (64GB RAM = full exact counts)
6. **Runner API**: Custom REST API on port 11435 (multi-model, streaming)
7. **RAG**: BM25 via tantivy + SQLite snippet store
8. **Primary languages**: TypeScript, Python, JavaScript/JSX, Rust, Shell (SQL deferred until SQL training phase)

---

## How to Resume Work

1. Read this file (`AGENT.md`) — project philosophy, constraints, architecture
2. Read `DECISIONS.md` — all design decisions and session context
3. Check `FEATURES.md` for the full feature list and what's done
4. Check the Current Status table above
5. Run `python3 scripts/monitor.py --full` — see data/model/build status at a glance
6. Run `cargo test --workspace` to verify everything compiles
7. Continue from the next incomplete phase

### Quick Monitor Commands

```bash
python3 scripts/monitor.py           # One-shot status dashboard
python3 scripts/monitor.py --watch   # Live refreshing (5s interval)
python3 scripts/monitor.py --full    # Include cargo check + test results
python3 scripts/monitor.py --json    # Machine-readable output
```

### GitHub

- **Repo**: [github.com/dancherbu/sllm](https://github.com/dancherbu/sllm) (public)
- **Branch protection**: `master` requires 1 PR review, admin self-approval enabled
- **Local path**: `~/Projects/sllm/`
