//! # RAG Module
//!
//! Retrieval-Augmented Generation — extends the brain with external knowledge.
//! Uses BM25 text search + SQLite storage for code snippet retrieval.
//!
//! This module is stubbed for Phase 1. Full implementation in Phase 4.

mod index;
mod retriever;
mod store;

pub use index::{RagIndex, InMemoryIndex};
pub use retriever::Retriever;
pub use store::{SnippetStore, InMemoryStore, Snippet};
