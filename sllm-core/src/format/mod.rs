//! # Format Module
//!
//! The `brain.sllm` binary file format — self-describing, mmap-compatible,
//! and designed for portability across devices.

mod header;
mod reader;
mod writer;

pub use header::BrainHeader;
pub use reader::BrainReader;
pub use writer::BrainWriter;

/// Errors from file format operations.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("invalid magic bytes (expected 'SLLM')")]
    InvalidMagic,
    #[error("unsupported format version: {0}")]
    UnsupportedVersion(u16),
    #[error("checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    #[error("truncated file data")]
    TruncatedData,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("brain deserialization error: {0}")]
    Brain(#[from] crate::brain::BrainError),
    #[error("tokenizer deserialization error: {0}")]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
}
