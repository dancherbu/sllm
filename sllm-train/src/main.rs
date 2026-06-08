//! sLLM Training Engine
//!
//! Streaming, gradient-free model trainer. Reads text/code data,
//! tokenizes it, and updates associative count tables in the brain.
//!
//! Usage:
//!   sllm-train --data ./data/ --output ./models/model.sllm --name my-model
//!   sllm-train --resume ./models/model.sllm --data ./more-data/

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use sllm_core::brain::NgramBrain;
use sllm_core::format::{BrainReader, BrainWriter};
use sllm_core::tokenizer::{BpeTokenizer, BpeTrainer};

/// sLLM Training Engine — gradient-free model trainer
#[derive(Parser, Debug)]
#[command(name = "sllm-train", about = "Train an sLLM model from text/code data")]
struct Args {
    /// Path to training data (file or directory)
    #[arg(short, long)]
    data: PathBuf,

    /// Output path for the trained model (.sllm file)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Model name (stored in the brain.sllm header)
    #[arg(short, long, default_value = "sllm-base")]
    name: String,

    /// Resume training from an existing model
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Target vocabulary size for BPE tokenizer
    #[arg(long, default_value = "16384")]
    vocab_size: usize,

    /// Maximum associations per n-gram table (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_associations: u64,

    /// Checkpoint interval (save every N files processed)
    #[arg(long, default_value = "100")]
    checkpoint_interval: usize,

    /// Minimum count threshold for pruning during consolidation
    #[arg(long, default_value = "0")]
    prune_threshold: u32,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sllm_train=info".into()),
        )
        .init();

    let args = Args::parse();

    // Determine output path
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./models/model.sllm"));

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load or create model
    let (mut tokenizer, mut brain) = if let Some(ref resume_path) = args.resume {
        info!("Resuming from: {}", resume_path.display());
        let loaded = BrainReader::read_owned(resume_path)
            .context("Failed to read existing model")?;
        info!(
            "Loaded model '{}': {} associations, {} tokens trained",
            loaded.header.model_name,
            loaded.brain.total_associations(),
            loaded.brain.tokens_trained()
        );
        (loaded.tokenizer, loaded.brain)
    } else {
        info!("Creating new model '{}'", args.name);
        info!("Building tokenizer from training data (vocab_size={})...", args.vocab_size);

        // Build tokenizer from training data
        let text_iter = collect_text_files(&args.data)?;
        let trainer = BpeTrainer::new(args.vocab_size);
        let tokenizer = trainer.train(text_iter.iter().map(|s| s.as_str()));
        info!("Tokenizer built: {} tokens, {} merges", tokenizer.vocab().len(), tokenizer.num_merges());

        let brain = NgramBrain::new(args.max_associations);
        (tokenizer, brain)
    };

    // Train on data files
    let files = list_text_files(&args.data)?;
    info!("Found {} training files", files.len());

    let mut files_processed = 0u64;
    let start_time = std::time::Instant::now();

    for (i, file_path) in files.iter().enumerate() {
        let text = match std::fs::read_to_string(file_path) {
            Ok(t) => t,
            Err(e) => {
                warn!("Skipping {}: {}", file_path.display(), e);
                continue;
            }
        };

        // Tokenize and train
        let tokens = tokenizer.encode(&text);
        if tokens.len() >= 5 {
            brain.train_sequence(&tokens);
        }

        files_processed += 1;

        // Progress logging
        if files_processed % 50 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = files_processed as f64 / elapsed;
            info!(
                "[{}/{}] {} associations | {} tokens trained | {:.1} files/sec",
                i + 1,
                files.len(),
                brain.total_associations(),
                brain.tokens_trained(),
                rate,
            );
        }

        // Checkpoint
        if args.checkpoint_interval > 0 && files_processed as usize % args.checkpoint_interval == 0
        {
            info!("Checkpointing to {}...", output_path.display());
            BrainWriter::write(&output_path, &args.name, &tokenizer, &brain)?;
        }
    }

    // Consolidation pass (prune low-count associations)
    if args.prune_threshold > 0 {
        let before = brain.total_associations();
        brain.prune_all(args.prune_threshold);
        let after = brain.total_associations();
        info!(
            "Consolidation: pruned {} associations (threshold={})",
            before - after,
            args.prune_threshold
        );
    }

    // Final save
    info!("Saving model to {}...", output_path.display());
    BrainWriter::write(&output_path, &args.name, &tokenizer, &brain)?;

    let elapsed = start_time.elapsed();
    info!(
        "Training complete: {} files, {} associations, {} tokens in {:.1}s",
        files_processed,
        brain.total_associations(),
        brain.tokens_trained(),
        elapsed.as_secs_f64(),
    );

    Ok(())
}

/// Collect all text from files for tokenizer training.
fn collect_text_files(path: &Path) -> Result<Vec<String>> {
    let mut texts = Vec::new();

    if path.is_file() {
        let text = std::fs::read_to_string(path)?;
        texts.push(text);
    } else if path.is_dir() {
        for entry in walkdir(path)? {
            if let Ok(text) = std::fs::read_to_string(&entry) {
                // Limit per-file text for tokenizer training to avoid memory issues
                if text.len() < 1_000_000 {
                    texts.push(text);
                }
            }
        }
    }

    Ok(texts)
}

/// List all text/code files in a path recursively.
fn list_text_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        files = walkdir(path)?;
    }

    Ok(files)
}

/// Recursively walk a directory and return all file paths.
fn walkdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    fn walk_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories and common non-text dirs
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.starts_with('.') && name != "node_modules" && name != "target" && name != "__pycache__" {
                    walk_recursive(&path, files)?;
                }
            } else if path.is_file() {
                // Filter by common text/code extensions
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if matches!(
                        ext.as_str(),
                        "txt" | "md" | "py" | "rs" | "js" | "ts" | "jsx" | "tsx"
                            | "c" | "h" | "cpp" | "hpp" | "go" | "java" | "rb"
                            | "sh" | "bash" | "zsh" | "toml" | "yaml" | "yml"
                            | "json" | "html" | "css" | "sql"
                    ) {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    walk_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}
