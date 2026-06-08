//! # Tokenizer Module
//!
//! BPE (Byte-Pair Encoding) tokenizer with code-aware splitting.
//! Designed for a 16,384 token vocabulary optimized for Python, JS/TS, and Rust.

mod bpe;
mod vocab;

pub use bpe::{BpeTokenizer, BpeTrainer, TokenizerError};
pub use vocab::{SpecialToken, Vocab, VocabError};
