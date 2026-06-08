//! The `brain.sllm` file header.
//!
//! Contains model metadata, section offsets, and a CRC32 checksum.
//! Designed for forward compatibility (versioned, with section offsets).

use crate::{SLLM_MAGIC, SLLM_VERSION};

/// Header for a `brain.sllm` file.
#[derive(Debug, Clone)]
pub struct BrainHeader {
    /// File format version
    pub version: u16,
    /// Human-readable model name
    pub model_name: String,
    /// Vocabulary size (number of BPE tokens)
    pub vocab_size: u32,
    /// Context window size (tokens)
    pub context_window: u32,
    /// Maximum n-gram order (e.g., 5 for up to 5-gram)
    pub ngram_orders: u8,
    /// Total number of associations stored across all tables
    pub total_associations: u64,
    /// Total training tokens processed
    pub training_tokens_seen: u64,
    /// Unix timestamp when model was created/last saved
    pub created_at: u64,
    /// CRC32 checksum of the data sections (vocab + brain)
    pub data_checksum: u32,
    /// Byte offset where the tokenizer section begins
    pub tokenizer_offset: u64,
    /// Byte offset where the brain section begins
    pub brain_offset: u64,
}

impl BrainHeader {
    /// Total fixed header size in bytes (magic + version + fixed fields).
    /// The model name is variable-length, so actual header size varies.
    pub fn fixed_size() -> usize {
        4   // magic
        + 2 // version
        + 4 // header_size (u32, total including name)
        + 4 // model_name length (u32)
        // + model_name bytes (variable)
        + 4 // vocab_size
        + 4 // context_window
        + 1 // ngram_orders
        + 8 // total_associations
        + 8 // training_tokens_seen
        + 8 // created_at
        + 4 // data_checksum
        + 8 // tokenizer_offset
        + 8 // brain_offset
    }

    /// Actual header size including the variable-length model name.
    pub fn actual_size(&self) -> usize {
        Self::fixed_size() + self.model_name.len()
    }

    /// Serialize the header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.actual_size());

        // Magic
        buf.extend_from_slice(SLLM_MAGIC);

        // Version
        buf.extend_from_slice(&self.version.to_le_bytes());

        // Header size (total including name)
        let header_size = self.actual_size() as u32;
        buf.extend_from_slice(&header_size.to_le_bytes());

        // Model name (length-prefixed)
        let name_bytes = self.model_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        // Fixed fields
        buf.extend_from_slice(&self.vocab_size.to_le_bytes());
        buf.extend_from_slice(&self.context_window.to_le_bytes());
        buf.push(self.ngram_orders);
        buf.extend_from_slice(&self.total_associations.to_le_bytes());
        buf.extend_from_slice(&self.training_tokens_seen.to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.data_checksum.to_le_bytes());
        buf.extend_from_slice(&self.tokenizer_offset.to_le_bytes());
        buf.extend_from_slice(&self.brain_offset.to_le_bytes());

        buf
    }

    /// Deserialize a header from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, super::FormatError> {
        let mut pos = 0;

        // Magic
        if data.len() < 4 || &data[0..4] != SLLM_MAGIC {
            return Err(super::FormatError::InvalidMagic);
        }
        pos += 4;

        // Version
        if pos + 2 > data.len() {
            return Err(super::FormatError::TruncatedData);
        }
        let version = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        pos += 2;

        if version > SLLM_VERSION {
            return Err(super::FormatError::UnsupportedVersion(version));
        }

        // Header size
        if pos + 4 > data.len() {
            return Err(super::FormatError::TruncatedData);
        }
        let _header_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        // Model name
        if pos + 4 > data.len() {
            return Err(super::FormatError::TruncatedData);
        }
        let name_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + name_len > data.len() {
            return Err(super::FormatError::TruncatedData);
        }
        let model_name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| super::FormatError::TruncatedData)?
            .to_string();
        pos += name_len;

        // Fixed fields
        let remaining = data.len() - pos;
        if remaining < 4 + 4 + 1 + 8 + 8 + 8 + 4 + 8 + 8 {
            return Err(super::FormatError::TruncatedData);
        }

        let vocab_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let context_window = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let ngram_orders = data[pos];
        pos += 1;

        let total_associations = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let training_tokens_seen = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let created_at = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let data_checksum = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let tokenizer_offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let brain_offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());

        Ok(Self {
            version,
            model_name,
            vocab_size,
            context_window,
            ngram_orders,
            total_associations,
            training_tokens_seen,
            created_at,
            data_checksum,
            tokenizer_offset,
            brain_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> BrainHeader {
        BrainHeader {
            version: SLLM_VERSION,
            model_name: "test-model".to_string(),
            vocab_size: 16384,
            context_window: 128,
            ngram_orders: 5,
            total_associations: 1_000_000,
            training_tokens_seen: 50_000_000,
            created_at: 1717804800,
            data_checksum: 0xDEADBEEF,
            tokenizer_offset: 256,
            brain_offset: 4096,
        }
    }

    #[test]
    fn test_header_roundtrip() {
        let header = sample_header();
        let bytes = header.to_bytes();
        let restored = BrainHeader::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version, header.version);
        assert_eq!(restored.model_name, header.model_name);
        assert_eq!(restored.vocab_size, header.vocab_size);
        assert_eq!(restored.context_window, header.context_window);
        assert_eq!(restored.ngram_orders, header.ngram_orders);
        assert_eq!(restored.total_associations, header.total_associations);
        assert_eq!(restored.training_tokens_seen, header.training_tokens_seen);
        assert_eq!(restored.data_checksum, header.data_checksum);
        assert_eq!(restored.tokenizer_offset, header.tokenizer_offset);
        assert_eq!(restored.brain_offset, header.brain_offset);
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = b"NOPE";
        let result = BrainHeader::from_bytes(bytes);
        assert!(matches!(result, Err(super::super::FormatError::InvalidMagic)));
    }

    #[test]
    fn test_truncated_data() {
        let bytes = b"SLLM\x01";
        let result = BrainHeader::from_bytes(bytes);
        assert!(matches!(result, Err(super::super::FormatError::TruncatedData)));
    }
}
