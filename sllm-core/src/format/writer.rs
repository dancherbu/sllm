//! Writer for the `brain.sllm` file format.
//!
//! Serializes a trained model (tokenizer + brain) to a single portable file.
//! Supports atomic writes (write to `.tmp`, rename on completion).

use super::header::BrainHeader;
use super::FormatError;
use crate::brain::NgramBrain;
use crate::tokenizer::BpeTokenizer;
use crate::{SLLM_VERSION, DEFAULT_CONTEXT_WINDOW};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Writes a trained sLLM model to a `brain.sllm` file.
pub struct BrainWriter;

impl BrainWriter {
    /// Write a model to disk.
    ///
    /// Performs an atomic write: data is written to a temporary file first,
    /// then renamed to the final path on success.
    pub fn write(
        path: &Path,
        model_name: &str,
        tokenizer: &BpeTokenizer,
        brain: &NgramBrain,
    ) -> Result<(), FormatError> {
        let tmp_path = path.with_extension("sllm.tmp");

        // Serialize tokenizer and brain
        let tokenizer_bytes = tokenizer.to_bytes();
        let brain_bytes = brain.to_bytes();

        // Compute CRC32 over data sections
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&tokenizer_bytes);
        hasher.update(&brain_bytes);
        let checksum = hasher.finalize();

        // Build header (we need to know offsets first)
        let mut header = BrainHeader {
            version: SLLM_VERSION,
            model_name: model_name.to_string(),
            vocab_size: tokenizer.vocab().len() as u32,
            context_window: DEFAULT_CONTEXT_WINDOW,
            ngram_orders: 5,
            total_associations: brain.total_associations(),
            training_tokens_seen: brain.tokens_trained(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data_checksum: checksum,
            tokenizer_offset: 0, // Will be set below
            brain_offset: 0,     // Will be set below
        };

        let header_size = header.actual_size() as u64;
        header.tokenizer_offset = header_size;
        header.brain_offset = header_size + tokenizer_bytes.len() as u64;

        // Write to temp file
        let header_bytes = header.to_bytes();
        let total_size = header_bytes.len() + tokenizer_bytes.len() + brain_bytes.len();
        let mut file_data = Vec::with_capacity(total_size);
        file_data.extend_from_slice(&header_bytes);
        file_data.extend_from_slice(&tokenizer_bytes);
        file_data.extend_from_slice(&brain_bytes);

        fs::write(&tmp_path, &file_data)?;

        // Atomic rename
        fs::rename(&tmp_path, path)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::NgramBrain;
    use crate::tokenizer::BpeTrainer;

    #[test]
    fn test_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sllm");

        let trainer = BpeTrainer::new(20);
        let tokenizer = trainer.train(["hello world", "hello there"].iter());
        let mut brain = NgramBrain::new(0);
        brain.train_sequence(&tokenizer.encode("hello world"));

        BrainWriter::write(&path, "test-model", &tokenizer, &brain).unwrap();
        assert!(path.exists());

        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[0..4], b"SLLM");
    }
}
