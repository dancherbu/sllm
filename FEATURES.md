# FEATURES.md — sLLM Feature Roadmap

> Complete feature inventory: what's built, what's next, what's possible.
> Last updated: 2026-06-08

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ✅ | Complete and tested |
| 🔨 | In progress |
| 📋 | Planned (approved, in roadmap) |
| 💡 | Future extension (designed, not yet scheduled) |
| 🔬 | Research / experimental (may or may not happen) |

---

## Phase 1: Foundation

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 1.1 | Cargo workspace setup | 📋 | Multi-crate workspace: sllm-core, sllm-train, sllm-run, sllm-cli |
| 1.2 | `brain.sllm` file format | 📋 | Self-describing binary format with magic, version, sections, CRC32 |
| 1.3 | File format reader (mmap) | 📋 | Zero-copy memory-mapped reader via `memmap2` |
| 1.4 | File format writer (streaming) | 📋 | Streaming serializer with atomic checkpoint writes |
| 1.5 | BPE tokenizer — training | 📋 | Learn BPE merges from a text/code corpus |
| 1.6 | BPE tokenizer — encode/decode | 📋 | Convert text ↔ token IDs, code-aware (preserves indentation) |
| 1.7 | Special tokens | 📋 | `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<SEP>`, `<THINK>`, `<CODE>`, `<TOOL>` |
| 1.8 | Bigram count table | 📋 | Conditional counts: P(B \| A) via HashMap |
| 1.9 | Trigram count table | 📋 | Conditional counts: P(C \| A, B) |
| 1.10 | 4-gram + 5-gram tables | 📋 | Higher-order n-gram tables |
| 1.11 | Count-Min Sketch | 📋 | Memory-efficient approximate counting for lower n-grams |
| 1.12 | Interpolated smoothing | 📋 | Weighted combination of n-gram orders for prediction |
| 1.13 | Basic token sampling | 📋 | Greedy, top-k, top-p, temperature |

---

## Phase 2: Training Pipeline

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 2.1 | Streaming data pipeline | 📋 | Line-by-line file reading, never full dataset in memory |
| 2.2 | Parallel file processing | 📋 | Rayon-based parallel ingestion of multiple files |
| 2.3 | Training CLI | 📋 | `sllm-train --data <path> --output <path> --phase <phase>` |
| 2.4 | Phase 1: English curriculum | 📋 | TinyStories / Gutenberg training for base language patterns |
| 2.5 | Phase 2: Code syntax | 📋 | Python, JS/TS, Rust code — learn keywords, brackets, indentation |
| 2.6 | Phase 3: Code semantics | 📋 | Functions, classes, modules — learn structure and composition |
| 2.7 | Periodic checkpointing | 📋 | Auto-save model to disk at configurable intervals |
| 2.8 | Resumable training | 📋 | `--resume <model.sllm>` to continue from a checkpoint |
| 2.9 | Consolidation / "sleep" pass | 📋 | Prune low-count associations, boost high-MI paths |
| 2.10 | Count quantization | 📋 | Log-bucket compress counts (32-bit → 8-bit) post-training |
| 2.11 | Training progress metrics | 📋 | Live: tokens/sec, associations, memory usage, perplexity estimate |
| 2.12 | Data deduplication | 📋 | Skip near-duplicate text blocks to prevent over-counting |

---

## Phase 3: Inference Runner

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 3.1 | mmap model loader | 📋 | Zero-copy model loading via `memmap2::Mmap` |
| 3.2 | HTTP API server | 📋 | Axum-based server on port 11435 |
| 3.3 | `POST /v1/generate` | 📋 | Text generation endpoint with streaming (SSE) |
| 3.4 | `GET /v1/models` | 📋 | List loaded models with metadata |
| 3.5 | `POST /v1/models/load` | 📋 | Dynamically load a model from disk |
| 3.6 | `POST /v1/models/unload` | 📋 | Unload a model to free memory |
| 3.7 | `GET /v1/health` | 📋 | Server health check |
| 3.8 | `POST /v1/tokenize` | 📋 | Tokenize input text (debugging/inspection) |
| 3.9 | Model registry | 📋 | Scan models directory, read headers, list available models |
| 3.10 | Multi-model serving | 📋 | Load and serve multiple models simultaneously |
| 3.11 | Repetition penalty | 📋 | Reduce probability of recently generated tokens |
| 3.12 | Stop sequences | 📋 | Stop generation at specific token patterns |

---

## Phase 4: RAG + CLI

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 4.1 | BM25 text index | 📋 | Full-text search index for code snippets (tantivy or custom) |
| 4.2 | SQLite snippet store | 📋 | Persistent storage for indexed code/doc snippets |
| 4.3 | RAG retriever | 📋 | Top-k retrieval + score fusion with brain lookups |
| 4.4 | RAG-conditioned training | 📋 | Retrieved context conditions count updates during training |
| 4.5 | `sllm list` | 📋 | List available models with metadata |
| 4.6 | `sllm run <model>` | 📋 | Interactive chat session |
| 4.7 | `sllm serve` | 📋 | Start the inference API server |
| 4.8 | `sllm train` | 📋 | Invoke training with CLI arguments |
| 4.9 | `sllm info <model>` | 📋 | Display model metadata, training stats |
| 4.10 | `sllm merge` | 📋 | Merge two models (additive count combination) |
| 4.11 | `sllm export` | 📋 | Export model data to JSON for inspection |
| 4.12 | Code snippet indexer | 📋 | Auto-index a codebase directory for RAG |

---

## Phase 5: Hyperdimensional Computing (HDC) Upgrade

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 5.1 | HDC token embeddings | 💡 | Encode tokens as high-dimensional sparse binary vectors (10k–64k dims) |
| 5.2 | HDC binding (XOR) | 💡 | Compose context vectors via bitwise XOR for positional encoding |
| 5.3 | HDC bundling (majority) | 💡 | Combine vectors via majority vote for set representation |
| 5.4 | Hybrid brain | 💡 | Count tables + HDC projections running in parallel |
| 5.5 | HDC similarity search | 💡 | Hamming distance for fast approximate nearest-neighbor lookup |
| 5.6 | HDC model format | 💡 | Extended `brain.sllm` sections for HDC vectors |

---

## Phase 6: Contrastive Hebbian Learning

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 6.1 | Positive/negative examples | 💡 | Strengthen real next-tokens, weaken corrupted/random tokens |
| 6.2 | Forward-Forward style updates | 💡 | Hinton-inspired layer-local contrastive learning |
| 6.3 | Corruption strategies | 💡 | Token swap, deletion, insertion for negative examples |
| 6.4 | Goodness scoring | 💡 | Per-layer "goodness" metric (sum of squared counts or similar) |
| 6.5 | Contrastive consolidation | 💡 | Sleep pass uses contrastive signal for pruning decisions |

---

## Phase 7: Agentic Coding Loop

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 7.1 | Special action tokens | 💡 | `<THINK>`, `<PLAN>`, `<CODE>`, `<COMPILE>`, `<TEST>`, `<FIX>`, `<TOOL>` |
| 7.2 | Trajectory training | 💡 | Train on sequences: think → plan → code → compile → observe → fix |
| 7.3 | Tool call generation | 💡 | Model generates `<TOOL:file_read>`, `<TOOL:compile>`, etc. |
| 7.4 | Observation feedback | 💡 | Feed compiler/test output back as training signal |
| 7.5 | File operations | 💡 | Runner can execute: read file, write file, list directory |
| 7.6 | Compile integration | 💡 | Runner can invoke compiler and capture output |
| 7.7 | Test execution | 💡 | Runner can run test suites and parse results |
| 7.8 | Multi-step planning | 💡 | Generate and follow multi-step coding plans |
| 7.9 | Self-correction loop | 💡 | Detect errors in own output and attempt fix |

---

## Phase 8: Modular Experts

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 8.1 | Expert modules | 💡 | Separate brain sub-tables: syntax, semantics, planning, personal style |
| 8.2 | Hash-based router | 💡 | Cheap routing: hash input context → select active experts |
| 8.3 | Expert training | 💡 | Train each expert on domain-specific data |
| 8.4 | Expert merging | 💡 | Combine expert outputs via weighted voting |
| 8.5 | Hot-swappable experts | 💡 | Load/unload expert modules at runtime |
| 8.6 | Personal style expert | 💡 | Dedicated module that learns YOUR coding patterns and preferences |

---

## Future Extensions

### Memory & Context

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.1 | Hierarchical memory | 💡 | Short-term (ring buffer) + medium (trie/graph) + long (brain.sllm) |
| F.2 | Sliding KV cache | 💡 | Tiny key-value cache with hash-similarity eviction |
| F.3 | Context extension to 512+ | 💡 | Extend context window beyond 128 tokens |
| F.4 | Conversation memory | 💡 | Multi-turn context persistence across chat sessions |
| F.5 | Session state snapshots | 💡 | Save/restore conversation state |

### Model Management

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.6 | Model versioning | 💡 | Semantic versioning in brain.sllm header |
| F.7 | Differential updates | 💡 | Delta files containing only new/changed associations |
| F.8 | Model merging (federated) | 💡 | Merge models from different devices (additive counts + decay) |
| F.9 | Model marketplace | 🔬 | Push/pull models from a registry (like Docker Hub for brains) |
| F.10 | Model compression | 💡 | Post-training quantization of counts to 4-bit |
| F.11 | Model pruning | 💡 | Remove low-utility associations below threshold |
| F.12 | Model splitting | 💡 | Split a model into domain-specific sub-models |

### Training Enhancements

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.13 | Web scraper ingestion | 💡 | Autonomous internet learning with text extraction |
| F.14 | Live codebase watcher | 💡 | inotify-based file watcher that retrains on code changes |
| F.15 | Git history training | 💡 | Learn from commit diffs and code evolution patterns |
| F.16 | Synthetic data generation | 💡 | Use a stronger model to generate training trajectories |
| F.17 | Distillation bootstrap | 🔬 | One-time distillation from a neural SLM into associative format |
| F.18 | Mutual information estimator | 💡 | Better pruning decisions based on MI between associations |
| F.19 | Decay functions | 💡 | Time-based count decay (newer data weighted more) |
| F.20 | Curriculum auto-detection | 💡 | Automatically detect training phase from data characteristics |

### Deployment & Platform

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.21 | Cross-compilation targets | 💡 | Build for: x86_64, aarch64 (ARM), RISC-V |
| F.22 | Android build | 💡 | NDK cross-compilation for Android devices |
| F.23 | iOS build | 🔬 | Static library for iOS via cargo-lipo |
| F.24 | WebAssembly (WASM) | 💡 | Compile runner to WASM for browser-based inference |
| F.25 | Raspberry Pi support | 💡 | Optimized build for ARM Cortex-A (Pi 4/5) |
| F.26 | Docker image | 💡 | Lightweight container for server deployment |
| F.27 | Systemd service | 💡 | Auto-start runner as a system service |
| F.28 | SIMD optimization | 💡 | x86 SSE/AVX and ARM NEON intrinsics for hot paths |

### IDE & Editor Integration

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.29 | LSP server | 💡 | Language Server Protocol for real-time autocomplete |
| F.30 | VSCode extension | 💡 | Inline completions powered by sLLM |
| F.31 | Neovim plugin | 💡 | Lua plugin for sLLM completions |
| F.32 | JetBrains plugin | 🔬 | IntelliJ/PyCharm integration |
| F.33 | Helix integration | 💡 | LSP-based integration with Helix editor |
| F.34 | Zed integration | 💡 | Zed editor inline completion |

### API & Compatibility

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.35 | OpenAI-compatible API | 💡 | `/v1/chat/completions` endpoint for drop-in compatibility |
| F.36 | Ollama-compatible API | 💡 | `/api/generate`, `/api/chat` for Ollama client compatibility |
| F.37 | gRPC API | 🔬 | High-performance gRPC interface for production |
| F.38 | WebSocket streaming | 💡 | Bidirectional streaming for real-time applications |
| F.39 | Batch generation | 💡 | Process multiple prompts in parallel |
| F.40 | Embeddings endpoint | 💡 | Generate token/context embeddings (HDC-based) |

### Observability & Debugging

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.41 | Token probability inspector | 💡 | Show top-k next-token probabilities for a given context |
| F.42 | Association visualizer | 💡 | Web UI showing n-gram associations as a graph |
| F.43 | Training dashboard | 💡 | Live metrics: tokens/sec, perplexity, memory, associations |
| F.44 | Model diff tool | 💡 | Compare two brain.sllm files (what changed?) |
| F.45 | Benchmark suite | 💡 | Pass@1, perplexity, coherence on standard coding problems |
| F.46 | Power/latency profiler | 💡 | Measure real watts and latency on target devices |

### Advanced Brain Architectures

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.47 | DAWG / Trie storage | 💡 | Directed Acyclic Word Graph for prefix compression |
| F.48 | LSH bucketing | 💡 | Locality-Sensitive Hashing for approximate context matching |
| F.49 | State Space hints | 🔬 | Mamba/RWKV-style linear recurrences (count-approximated) |
| F.50 | Graph neural memory | 🔬 | Code dependency graph stored alongside n-grams |
| F.51 | Attention-like routing | 🔬 | Lightweight dot-product routing over sparse hashes |

### Security & Privacy

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.52 | Differential privacy | 🔬 | Noise injection to prevent memorization of sensitive code |
| F.53 | Encrypted brain files | 💡 | AES-256 encryption for brain.sllm |
| F.54 | Access control | 💡 | API key authentication for the runner |
| F.55 | Safety filters | 💡 | Prevent generation of known-bad patterns |
| F.56 | Audit logging | 💡 | Log all training data sources and model mutations |

### Personalization

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F.57 | Personal coding style learning | 💡 | Dedicated expert that learns your naming, formatting, patterns |
| F.58 | Project-specific fine-tuning | 💡 | Train a model layer specifically for a single project |
| F.59 | User preference profiles | 💡 | Store user settings (temperature, style, language preference) |
| F.60 | Code review suggestions | 💡 | Generate suggestions based on your past review patterns |
| F.61 | Commit message generation | 💡 | Learn your commit message style and generate them |

---

## Feature Count Summary

| Category | ✅ Done | 🔨 WIP | 📋 Planned | 💡 Future | 🔬 Research | Total |
|----------|---------|--------|-----------|-----------|-------------|-------|
| Phase 1: Foundation | 0 | 0 | 13 | 0 | 0 | 13 |
| Phase 2: Training | 0 | 0 | 12 | 0 | 0 | 12 |
| Phase 3: Runner | 0 | 0 | 12 | 0 | 0 | 12 |
| Phase 4: RAG + CLI | 0 | 0 | 12 | 0 | 0 | 12 |
| Phase 5: HDC | 0 | 0 | 0 | 6 | 0 | 6 |
| Phase 6: Contrastive | 0 | 0 | 0 | 5 | 0 | 5 |
| Phase 7: Agentic | 0 | 0 | 0 | 9 | 0 | 9 |
| Phase 8: Experts | 0 | 0 | 0 | 6 | 0 | 6 |
| Future Extensions | 0 | 0 | 0 | 48 | 13 | 61 |
| **TOTAL** | **0** | **0** | **49** | **74** | **13** | **136** |
