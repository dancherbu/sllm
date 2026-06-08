//! BPE (Byte-Pair Encoding) tokenizer.
//!
//! Learns merge rules from a text corpus and uses them to encode/decode text.
//! Code-aware: preserves indentation, brackets, and operators as atomic tokens.

use super::vocab::{SpecialToken, Vocab};
use std::collections::HashMap;

/// A single BPE merge rule: combine two tokens into one.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeRule {
    /// First token in the pair
    pub left: String,
    /// Second token in the pair
    pub right: String,
    /// Resulting merged token
    pub merged: String,
}

/// Trains a BPE vocabulary from text data.
#[derive(Debug)]
pub struct BpeTrainer {
    target_vocab_size: usize,
}

impl BpeTrainer {
    /// Create a new BPE trainer targeting the given vocabulary size.
    pub fn new(target_vocab_size: usize) -> Self {
        Self { target_vocab_size }
    }

    /// Train BPE merges from an iterator of text chunks.
    ///
    /// Returns a trained `BpeTokenizer` ready for encoding/decoding.
    pub fn train<I, S>(&self, text_iter: I) -> BpeTokenizer
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut vocab = Vocab::new();
        let mut merge_rules: Vec<MergeRule> = Vec::new();

        // Phase 1: Build initial character-level vocabulary from all text
        let mut word_freqs: HashMap<Vec<String>, u64> = HashMap::new();

        for text in text_iter {
            let text = text.as_ref();
            for word in pre_tokenize(text) {
                // Split word into individual characters (with word-boundary markers)
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                if !chars.is_empty() {
                    *word_freqs.entry(chars).or_insert(0) += 1;
                }
            }
        }

        // Add all individual characters to vocab
        let mut char_set: Vec<String> = Vec::new();
        for word in word_freqs.keys() {
            for ch in word {
                if !char_set.contains(ch) {
                    char_set.push(ch.clone());
                }
            }
        }
        char_set.sort();
        for ch in &char_set {
            vocab.add_token(ch);
        }

        // Phase 2: Iteratively merge the most frequent pair
        while vocab.len() < self.target_vocab_size {
            // Count all adjacent pairs
            let mut pair_counts: HashMap<(String, String), u64> = HashMap::new();
            for (word, freq) in &word_freqs {
                for pair in word.windows(2) {
                    *pair_counts
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_insert(0) += freq;
                }
            }

            // Find the most frequent pair
            let best_pair = pair_counts.into_iter().max_by_key(|&(_, count)| count);

            let Some(((left, right), _count)) = best_pair else {
                break; // No more pairs to merge
            };

            let merged = format!("{}{}", left, right);
            vocab.add_token(&merged);

            merge_rules.push(MergeRule {
                left: left.clone(),
                right: right.clone(),
                merged: merged.clone(),
            });

            // Apply this merge to all words in the frequency table
            let mut new_word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
            for (word, freq) in word_freqs {
                let merged_word = apply_merge(&word, &left, &right, &merged);
                *new_word_freqs.entry(merged_word).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;
        }

        BpeTokenizer {
            vocab,
            merge_rules,
        }
    }
}

/// Apply a single merge rule to a tokenized word.
fn apply_merge(word: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
    let mut result = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
            result.push(merged.to_string());
            i += 2;
        } else {
            result.push(word[i].clone());
            i += 1;
        }
    }
    result
}

/// Pre-tokenize text into words/chunks that BPE will further split.
/// Code-aware: preserves indentation, operators, brackets, and newlines.
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // Whitespace boundaries
            ' ' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                // Count consecutive spaces (preserve indentation)
                let mut spaces = String::from(" ");
                while chars.peek() == Some(&' ') {
                    spaces.push(' ');
                    chars.next();
                }
                tokens.push(spaces);
            }
            '\n' | '\r' | '\t' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(c.to_string());
            }
            // Programming punctuation: each is its own token
            '(' | ')' | '[' | ']' | '{' | '}' | ';' | ',' | ':' | '.' | '=' | '+' | '-'
            | '*' | '/' | '<' | '>' | '!' | '&' | '|' | '^' | '~' | '%' | '@' | '#'
            | '?' | '\\' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(c.to_string());
            }
            // String delimiters
            '"' | '\'' | '`' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(c.to_string());
            }
            // Regular characters: accumulate
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// BPE tokenizer that can encode text to token IDs and decode back.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BpeTokenizer {
    /// The token vocabulary
    pub vocab: Vocab,
    /// Ordered list of merge rules (applied in priority order)
    merge_rules: Vec<MergeRule>,
}

impl BpeTokenizer {
    /// Encode text into a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut token_ids = Vec::new();

        for word in pre_tokenize(text) {
            // Start with character-level tokens
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merge rules in order
            for rule in &self.merge_rules {
                symbols = apply_merge(&symbols, &rule.left, &rule.right, &rule.merged);
            }

            // Convert to IDs
            for symbol in &symbols {
                token_ids.push(self.vocab.encode_token(symbol));
            }
        }

        token_ids
    }

    /// Decode a sequence of token IDs back into text.
    pub fn decode(&self, token_ids: &[u16]) -> String {
        let mut text = String::new();
        for &id in token_ids {
            if let Some(token) = self.vocab.id_to_token(id) {
                // Skip special tokens in output
                let is_special = SpecialToken::all().iter().any(|s| s.as_str() == token);
                if !is_special {
                    text.push_str(token);
                }
            }
        }
        text
    }

    /// Returns the vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Returns the number of merge rules learned.
    pub fn num_merges(&self) -> usize {
        self.merge_rules.len()
    }

    /// Serialize the tokenizer (vocab + merges) to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.vocab.to_bytes();

        // Number of merge rules
        let num_merges = self.merge_rules.len() as u32;
        buf.extend_from_slice(&num_merges.to_le_bytes());

        // Each merge rule: left_len(u16) + left + right_len(u16) + right + merged_len(u16) + merged
        for rule in &self.merge_rules {
            for s in [&rule.left, &rule.right, &rule.merged] {
                let bytes = s.as_bytes();
                let len = bytes.len() as u16;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(bytes);
            }
        }

        buf
    }

    /// Deserialize a tokenizer from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), TokenizerError> {
        let (vocab, mut pos) = Vocab::from_bytes(data).map_err(TokenizerError::Vocab)?;

        if pos + 4 > data.len() {
            return Err(TokenizerError::TruncatedData);
        }
        let num_merges =
            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut merge_rules = Vec::with_capacity(num_merges);
        for _ in 0..num_merges {
            let mut strings = Vec::with_capacity(3);
            for _ in 0..3 {
                if pos + 2 > data.len() {
                    return Err(TokenizerError::TruncatedData);
                }
                let len =
                    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
                pos += 2;
                if pos + len > data.len() {
                    return Err(TokenizerError::TruncatedData);
                }
                let s = std::str::from_utf8(&data[pos..pos + len])
                    .map_err(|_| TokenizerError::InvalidUtf8)?
                    .to_string();
                pos += len;
                strings.push(s);
            }
            merge_rules.push(MergeRule {
                left: strings[0].clone(),
                right: strings[1].clone(),
                merged: strings[2].clone(),
            });
        }

        Ok((
            Self {
                vocab,
                merge_rules,
            },
            pos,
        ))
    }
}

/// Errors from tokenizer operations.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("vocabulary error: {0}")]
    Vocab(#[from] super::vocab::VocabError),
    #[error("tokenizer data is truncated")]
    TruncatedData,
    #[error("invalid UTF-8 in tokenizer data")]
    InvalidUtf8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_tokenize_basic() {
        let tokens = pre_tokenize("hello world");
        assert_eq!(tokens, vec!["hello", " ", "world"]);
    }

    #[test]
    fn test_pre_tokenize_code() {
        let tokens = pre_tokenize("def foo(x):");
        assert!(tokens.contains(&"def".to_string()));
        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&")".to_string()));
        assert!(tokens.contains(&":".to_string()));
    }

    #[test]
    fn test_pre_tokenize_indentation() {
        let tokens = pre_tokenize("    return x");
        // 4 spaces should be preserved as one token
        assert_eq!(tokens[0], "    ");
    }

    #[test]
    fn test_bpe_train_tiny() {
        let texts = vec!["the cat sat", "the cat mat", "the dog sat"];
        let trainer = BpeTrainer::new(30);
        let tokenizer = trainer.train(texts.into_iter());
        assert!(tokenizer.vocab().len() > 8); // More than just specials
        assert!(tokenizer.num_merges() > 0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let texts = vec![
            "hello world",
            "the cat sat on the mat",
            "hello hello hello",
        ];
        let trainer = BpeTrainer::new(50);
        let tokenizer = trainer.train(texts.iter());

        let original = "hello world";
        let encoded = tokenizer.encode(original);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_tokenizer_serialization_roundtrip() {
        let texts = vec!["abc def", "abc ghi"];
        let trainer = BpeTrainer::new(30);
        let tokenizer = trainer.train(texts.into_iter());

        let bytes = tokenizer.to_bytes();
        let (restored, consumed) = BpeTokenizer::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.vocab().len(), tokenizer.vocab().len());
        assert_eq!(restored.num_merges(), tokenizer.num_merges());
    }
}
