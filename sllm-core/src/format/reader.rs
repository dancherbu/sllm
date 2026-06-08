//! Reader for the `brain.sllm` file format.
//!
//! Supports two modes:
//! - **Owned**: Read the entire file into memory (for the training engine)
//! - **Mapped**: Memory-map the file read-only (for the inference runner)

use super::header::BrainHeader;
use super::FormatError;
use crate::brain::NgramBrain;
use crate::tokenizer::BpeTokenizer;
use std::path::Path;

/// A loaded sLLM model (header + tokenizer + brain).
#[derive(Debug)]
pub struct LoadedModel {
    /// File header with metadata
    pub header: BrainHeader,
    /// The BPE tokenizer
    pub tokenizer: BpeTokenizer,
    /// The associative brain
    pub brain: NgramBrain,
}

/// Reads `brain.sllm` files.
pub struct BrainReader;

impl BrainReader {
    /// Read a model file into memory (owned mode).
    ///
    /// Use this in the training engine where you need mutable access.
    pub fn read_owned(path: &Path) -> Result<LoadedModel, FormatError> {
        let data = std::fs::read(path)?;
        Self::parse(&data)
    }

    /// Memory-map a model file (read-only mode).
    ///
    /// Use this in the inference runner for zero-copy access.
    ///
    /// # Safety
    /// The returned model borrows from the mmap, but we clone the data
    /// into owned structures for safety. True zero-copy would require
    /// unsafe lifetime gymnastics that we avoid for v1.
    pub fn read_mmap(path: &Path) -> Result<LoadedModel, FormatError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::parse(&mmap)
    }

    /// Parse a model from a byte slice (works for both owned and mmap data).
    fn parse(data: &[u8]) -> Result<LoadedModel, FormatError> {
        // Parse header
        let header = BrainHeader::from_bytes(data)?;

        // Verify CRC32 checksum
        let tokenizer_start = header.tokenizer_offset as usize;
        let data_section = &data[tokenizer_start..];

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data_section);
        let actual_checksum = hasher.finalize();

        // Note: We compute CRC over tokenizer+brain combined, but we wrote
        // them separately. We need to split and verify correctly.
        // For v1, we verify by re-parsing and checking the header checksum
        // matches what we'd compute. Skip strict verification for now and
        // just parse the sections.

        // Parse tokenizer
        let tokenizer_data = &data[tokenizer_start..];
        let (tokenizer, tokenizer_consumed) = BpeTokenizer::from_bytes(tokenizer_data)?;

        // Parse brain
        let brain_start = tokenizer_start + tokenizer_consumed;
        let brain_data = &data[brain_start..];
        let (brain, _brain_consumed) = NgramBrain::from_bytes(brain_data)?;

        // Verify checksum (compute over tokenizer + brain sections)
        let tokenizer_bytes = &data[tokenizer_start..brain_start];
        let brain_bytes_raw = &data[brain_start..];
        let mut check_hasher = crc32fast::Hasher::new();
        check_hasher.update(tokenizer_bytes);
        check_hasher.update(brain_bytes_raw);
        let computed = check_hasher.finalize();

        if computed != header.data_checksum {
            // Log warning but don't fail — allows forward compatibility
            // with minor format additions that append data.
            eprintln!(
                "Warning: checksum mismatch (expected {:#010x}, got {:#010x}). File may be corrupted or from a newer version.",
                header.data_checksum, computed
            );
        }

        // Suppress unused variable warning
        let _ = actual_checksum;

        Ok(LoadedModel {
            header,
            tokenizer,
            brain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::NgramBrain;
    use crate::format::BrainWriter;
    use crate::tokenizer::BpeTrainer;

    #[test]
    fn test_write_then_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.sllm");

        // Create model
        let trainer = BpeTrainer::new(25);
        let tokenizer = trainer.train(["the cat sat on the mat", "the dog ran"].iter());
        let mut brain = NgramBrain::new(0);
        let tokens = tokenizer.encode("the cat sat on the mat");
        brain.train_sequence(&tokens);

        // Write
        BrainWriter::write(&path, "roundtrip-test", &tokenizer, &brain).unwrap();

        // Read back (owned)
        let loaded = BrainReader::read_owned(&path).unwrap();
        assert_eq!(loaded.header.model_name, "roundtrip-test");
        assert_eq!(loaded.header.vocab_size, tokenizer.vocab().len() as u32);
        assert_eq!(loaded.tokenizer.vocab().len(), tokenizer.vocab().len());
        assert_eq!(
            loaded.brain.total_associations(),
            brain.total_associations()
        );
        assert_eq!(loaded.brain.tokens_trained(), brain.tokens_trained());

        // Read back (mmap)
        let loaded_mmap = BrainReader::read_mmap(&path).unwrap();
        assert_eq!(loaded_mmap.header.model_name, "roundtrip-test");
    }
}
