//! # sLLM Core
//!
//! The shared library for the sLLM (shallow Large Language Model) system.
//!
//! This crate provides:
//! - **Tokenizer**: BPE tokenizer with code-aware splitting (20k-24k multilingual vocab)
//! - **Brain**: Hebbian associative count tables (zero calculus, gradient-free)
//! - **Format**: `brain.sllm` binary file format (mmap-compatible)
//! - **RAG**: Retrieval-Augmented Generation index and retrieval
//! - **Eval**: Perplexity, generation, and convergence metrics
//!
//! ## Philosophy
//!
//! Everything in this crate operates without calculus. Learning is done via
//! integer count increments on conditional probability tables. No gradients,
//! no backpropagation, no floating-point weights.

pub mod brain;
pub mod eval;
pub mod format;
pub mod rag;
pub mod tokenizer;

/// The magic bytes identifying a brain.sllm file.
pub const SLLM_MAGIC: &[u8; 4] = b"SLLM";

/// Current file format version.
pub const SLLM_VERSION: u16 = 1;

/// Default vocabulary size for the BPE tokenizer.
pub const DEFAULT_VOCAB_SIZE: u32 = 22_000;

/// Default context window size (number of tokens for n-gram lookups).
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128;

/// Default HTTP port for the inference runner.
pub const DEFAULT_PORT: u16 = 11435;
