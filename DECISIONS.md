# sLLM — Session Decisions & Design Record

> **Purpose**: This file captures all design decisions, rationale, and session context so that any future session (human or AI) can pick up exactly where we left off. Read this alongside `AGENT.md`.

---

## Session 1: Foundation & Data Strategy (2026-06-08)

### Design Decisions Made

#### Architecture
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | **Rust** | Zero-cost abstractions, no GC pauses, native SIMD, mmap support |
| Learning method | **Hebbian associative counting** (n-gram count tables) | No gradients, no GPU, CPU-only, instant training |
| N-gram orders | **2 through 5** simultaneously | Higher orders capture structure, lower orders ensure coverage |
| File format | **`brain.sllm`** with CRC32 | Single portable file, mmap-friendly, versioned |
| Runner API | **Custom HTTP on port 11435** (like Ollama) | Multi-model registry, streaming, no external deps |
| GPU policy | **Intentionally CPU-only** | RTX 3090 available but excluded — no CUDA dependency |

#### Languages & Training Data
| Decision | Choice | Rationale |
|----------|--------|-----------|
| First natural language | **Ashanti Twi** (not Akuapem) | User's mother tongue / heritage language |
| Second natural language | **English** | Must achieve full fluency over time |
| Primary code languages | **TypeScript > Python > JavaScript/JSX > Rust** | Matches user's actual `~/Projects/` usage (3.7M TS, 3.3M Python, 2.2M JS lines) |
| SQL | **Excluded until SQL phase added** | ~60M lines are DB dumps, not hand-written code |
| Tokenizer vocab size | **20k–24k tokens** | Expanded from 16k to cover Twi + English + code |
| Tokenizer type | **BPE, multilingual** | Must handle Twi special chars (Ɛ/ɛ, Ɔ/ɔ) + code tokens |

#### Hardware Target
| Spec | Value |
|------|-------|
| Dev machine | Minisforum MS-A2 w/ Ryzen 9 9955HX (16C/32T), 64GB DDR5, 1TB NVMe |
| Training RAM budget | ~8–12 GB peak (64 GB available) |
| Training time target | < 1 hour for full corpus on dev machine |
| Inference RAM budget | < 200 MB (edge deployment) |
| Model file size | ~50 MB (`brain.sllm`) |

### Training Data Sources (All Free)

#### Phase 0: Ashanti Twi 🇬🇭
| Dataset | HuggingFace ID | Rows |
|---------|---------------|------|
| Twi Sentences 320k | `ghananlpcommunity/twi_sentences_320k` | 320k |
| English-Twi Pairs 4M | `ghananlpcommunity/english-twi_sentence-pairs-4m` | 4M |
| Twi Sentiments 400k | `ghananlpcommunity/twi-sentiments-corpus-400k` | 400k |
| Twi Emotions 400k | `ghananlpcommunity/twi-emotions-corpus-400k` | 400k |
| Asante Twi Bible | `ghananlpcommunity/asante-twi-bible-speech-text` | ~20k |
| GooAQ Twi 2M | `ghananlpcommunity/gooaq-twi-2m` | 2.95M |
| Code 170k Twi | `ghananlpcommunity/Code-170k-twi` | 177k |
| Twi Reasoning 1k | `ghananlpcommunity/twi-llm-reasoning-dataset-1k` | ~1k |
| Ghana QA | `ghananlpcommunity/ghana-qa` | 3.5M |

#### Phase 1: English
| Dataset | Source | Size |
|---------|--------|------|
| TinyStories | `roneneldan/TinyStories` | 2.1M stories |
| Simple English Wikipedia | Wikimedia dump | 509k lines |

#### Phase 2: Code (Public)
| Dataset | Source | Languages |
|---------|--------|-----------|
| The Stack Processed V2 | `vinsblack/The_Stack_Processed-v2` | Python, TypeScript, JavaScript, Rust |

#### Phase 3: Personal Code
| Source | Files | Lines |
|--------|-------|-------|
| `~/Projects/` (8 projects) | 20,524 files | ~4.5M lines |

### English Fluency Strategy

> **Long-term goal**: The sLLM should understand and generate fluent English.

Current plan:
1. **Phase 1 foundation**: TinyStories (simple English patterns) + Wikipedia (factual English)
2. **Future expansion** (not yet built):
   - Add Project Gutenberg books (classic literature, 50–100 books)
   - Add OpenWebText or equivalent web crawl subset
   - Add conversational English datasets (DailyDialog, etc.)
   - Add technical English (documentation, READMEs, man pages)
3. **Continuous learning**: The training engine supports incremental training — add more English data over time without restarting from scratch
4. **Evaluation**: Build an English perplexity benchmark to track fluency over versions

### Open Items for Future Sessions

- [ ] Google Books Ngrams shortcut — import pre-counted English n-grams for instant English head start
- [ ] GPT-2 BPE vocabulary bootstrap — use OpenAI's vocab as starting point for our tokenizer
- [ ] Expand English corpus with Gutenberg, web text, conversational data
- [ ] Build perplexity benchmarks per language (Twi, English, per-code-language)
- [ ] Model versioning — track brain.sllm versions with training data provenance

### Repository

- **GitHub**: [github.com/dancherbu/sllm](https://github.com/dancherbu/sllm) (public)
- **Branch protection**: `master` requires 1 PR review, admin bypass enabled (self-approval allowed)
- **Local path**: `~/Projects/sllm/`

---

## How to Resume Work

1. Read `AGENT.md` — project philosophy, constraints, architecture
2. Read this file (`DECISIONS.md`) — all design decisions and context
3. Check `FEATURES.md` — full feature roadmap with status
4. Run `cargo test --workspace` — verify everything compiles
5. Check `data/` directory — see what training data is available
6. Continue from the next incomplete item
