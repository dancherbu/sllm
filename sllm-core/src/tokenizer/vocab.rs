//! Vocabulary management for the sLLM tokenizer.
//!
//! Handles token ID ↔ string mappings and special tokens.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Special tokens used by the sLLM system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialToken {
    /// Padding token (ID 0)
    Pad,
    /// Unknown token (ID 1)
    Unk,
    /// Beginning of sequence (ID 2)
    Bos,
    /// End of sequence (ID 3)
    Eos,
    /// Separator between segments (ID 4)
    Sep,
    /// Marks a "thinking" / reasoning block (ID 5)
    Think,
    /// Marks a code block (ID 6)
    Code,
    /// Marks a tool call (ID 7)
    Tool,
}

impl SpecialToken {
    /// Returns the token ID for this special token.
    pub fn id(self) -> u16 {
        match self {
            Self::Pad => 0,
            Self::Unk => 1,
            Self::Bos => 2,
            Self::Eos => 3,
            Self::Sep => 4,
            Self::Think => 5,
            Self::Code => 6,
            Self::Tool => 7,
        }
    }

    /// Returns the string representation of this special token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pad => "<PAD>",
            Self::Unk => "<UNK>",
            Self::Bos => "<BOS>",
            Self::Eos => "<EOS>",
            Self::Sep => "<SEP>",
            Self::Think => "<THINK>",
            Self::Code => "<CODE>",
            Self::Tool => "<TOOL>",
        }
    }

    /// All special tokens in order.
    pub fn all() -> &'static [SpecialToken] {
        &[
            Self::Pad,
            Self::Unk,
            Self::Bos,
            Self::Eos,
            Self::Sep,
            Self::Think,
            Self::Code,
            Self::Tool,
        ]
    }

    /// Number of reserved special token IDs.
    pub fn count() -> u16 {
        Self::all().len() as u16
    }
}

/// Token vocabulary: maps between token IDs and their string representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Token ID → string (includes special tokens at indices 0..8)
    id_to_token: Vec<String>,
    /// String → token ID (for fast encoding)
    token_to_id: HashMap<String, u16>,
}

impl Vocab {
    /// Create a new vocabulary with only special tokens.
    pub fn new() -> Self {
        let mut vocab = Self {
            id_to_token: Vec::new(),
            token_to_id: HashMap::new(),
        };

        // Add all special tokens
        for special in SpecialToken::all() {
            let id = vocab.id_to_token.len() as u16;
            debug_assert_eq!(id, special.id());
            let s = special.as_str().to_string();
            vocab.token_to_id.insert(s.clone(), id);
            vocab.id_to_token.push(s);
        }

        vocab
    }

    /// Add a token to the vocabulary. Returns the assigned ID.
    /// If the token already exists, returns its existing ID.
    pub fn add_token(&mut self, token: &str) -> u16 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u16;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        id
    }

    /// Look up a token by its string. Returns None if not in vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u16> {
        self.token_to_id.get(token).copied()
    }

    /// Look up a token string by its ID. Returns None if ID is out of range.
    pub fn id_to_token(&self, id: u16) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Returns the total number of tokens (including special tokens).
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Returns true if the vocabulary contains only special tokens.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.len() <= SpecialToken::count() as usize
    }

    /// Returns the UNK token ID.
    pub fn unk_id(&self) -> u16 {
        SpecialToken::Unk.id()
    }

    /// Encode a token string to its ID, returning UNK if not found.
    pub fn encode_token(&self, token: &str) -> u16 {
        self.token_to_id.get(token).copied().unwrap_or(self.unk_id())
    }

    /// Serialize vocabulary to bytes for embedding in the brain.sllm file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Number of tokens
        let len = self.id_to_token.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        // Each token: length (u16) + utf8 bytes
        for token in &self.id_to_token {
            let bytes = token.as_bytes();
            let token_len = bytes.len() as u16;
            buf.extend_from_slice(&token_len.to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        buf
    }

    /// Deserialize vocabulary from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), VocabError> {
        let mut pos = 0;

        if data.len() < 4 {
            return Err(VocabError::TruncatedData);
        }
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut vocab = Self {
            id_to_token: Vec::with_capacity(len),
            token_to_id: HashMap::with_capacity(len),
        };

        for _ in 0..len {
            if pos + 2 > data.len() {
                return Err(VocabError::TruncatedData);
            }
            let token_len =
                u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;

            if pos + token_len > data.len() {
                return Err(VocabError::TruncatedData);
            }
            let token = std::str::from_utf8(&data[pos..pos + token_len])
                .map_err(|_| VocabError::InvalidUtf8)?
                .to_string();
            pos += token_len;

            let id = vocab.id_to_token.len() as u16;
            vocab.token_to_id.insert(token.clone(), id);
            vocab.id_to_token.push(token);
        }

        Ok((vocab, pos))
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur when working with vocabulary data.
#[derive(Debug, thiserror::Error)]
pub enum VocabError {
    #[error("vocabulary data is truncated")]
    TruncatedData,
    #[error("invalid UTF-8 in token data")]
    InvalidUtf8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_ids() {
        assert_eq!(SpecialToken::Pad.id(), 0);
        assert_eq!(SpecialToken::Unk.id(), 1);
        assert_eq!(SpecialToken::Tool.id(), 7);
        assert_eq!(SpecialToken::count(), 8);
    }

    #[test]
    fn test_new_vocab_has_special_tokens() {
        let vocab = Vocab::new();
        assert_eq!(vocab.len(), 8);
        assert_eq!(vocab.id_to_token(0), Some("<PAD>"));
        assert_eq!(vocab.id_to_token(7), Some("<TOOL>"));
        assert!(vocab.is_empty()); // "empty" means only specials
    }

    #[test]
    fn test_add_and_lookup() {
        let mut vocab = Vocab::new();
        let id = vocab.add_token("def");
        assert_eq!(id, 8); // First non-special
        assert_eq!(vocab.token_to_id("def"), Some(8));
        assert_eq!(vocab.id_to_token(8), Some("def"));
    }

    #[test]
    fn test_duplicate_add_returns_same_id() {
        let mut vocab = Vocab::new();
        let id1 = vocab.add_token("for");
        let id2 = vocab.add_token("for");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_encode_unknown_returns_unk() {
        let vocab = Vocab::new();
        assert_eq!(vocab.encode_token("nonexistent"), SpecialToken::Unk.id());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut vocab = Vocab::new();
        vocab.add_token("def");
        vocab.add_token("return");
        vocab.add_token("if");

        let bytes = vocab.to_bytes();
        let (restored, consumed) = Vocab::from_bytes(&bytes).unwrap();

        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.len(), vocab.len());
        assert_eq!(restored.token_to_id("def"), Some(8));
        assert_eq!(restored.token_to_id("return"), Some(9));
        assert_eq!(restored.id_to_token(0), Some("<PAD>"));
    }
}
