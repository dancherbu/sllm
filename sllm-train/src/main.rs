//! sLLM Training Engine
//!
//! Streaming, gradient-free model trainer. Reads text/code data,
//! tokenizes it, and updates associative count tables in the brain.
//!
//! Usage:
//!   sllm-train --auto --data ./data/ --output ./models/brain.sllm --name sllm-v1
//!   sllm-train --data ./data/twi/ --output ./models/twi.sllm --name twi-only
//!   sllm-train --resume ./models/model.sllm --data ./more-data/

use anyhow::{Context, Result};
use clap::Parser;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

use sllm_core::brain::NgramBrain;
use sllm_core::format::{BrainReader, BrainWriter};
use sllm_core::tokenizer::{BpeTokenizer, BpeTrainer};

mod curriculum;
mod evaluator;
mod progress;

use curriculum::Curriculum;
use progress::{PhaseStatus, ProgressWriter, TrainingStatus};

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
    #[arg(short, long, default_value = "sllm-v1")]
    name: String,

    /// Resume training from an existing model
    #[arg(short, long)]
    resume: Option<PathBuf>,

    /// Fully autonomous training mode: runs all curriculum phases
    /// until convergence with self-evaluation between phases.
    #[arg(long)]
    auto: bool,

    /// Target vocabulary size for BPE tokenizer
    #[arg(long, default_value = "22000")]
    vocab_size: usize,

    /// Maximum associations per n-gram table (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_associations: u64,

    /// Checkpoint interval (save every N files processed)
    #[arg(long, default_value = "500")]
    checkpoint_interval: usize,

    /// Minimum count threshold for pruning during consolidation
    #[arg(long, default_value = "0")]
    prune_threshold: u32,

    /// Maximum lines per phase to sample for tokenizer training
    #[arg(long, default_value = "100000")]
    tokenizer_sample_lines: usize,
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

    // Set up graceful shutdown on SIGINT/SIGTERM
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        ctrlc::set_handler(move || {
            if shutdown.load(Ordering::Relaxed) {
                // Second interrupt — force exit
                std::process::exit(1);
            }
            info!("Received interrupt — saving checkpoint and exiting...");
            shutdown.store(true, Ordering::Relaxed);
        })
        .expect("Failed to set Ctrl-C handler");
    }

    if args.auto {
        run_autonomous(&args, &shutdown)
    } else {
        run_single_phase(&args, &shutdown)
    }
}

/// Run fully autonomous multi-phase training.
fn run_autonomous(args: &Args, shutdown: &Arc<AtomicBool>) -> Result<()> {
    info!("═══════════════════════════════════════════════════════");
    info!("  sLLM Autonomous Training");
    info!("  Data: {}", args.data.display());
    info!("  Vocab: {} tokens", args.vocab_size);
    info!("═══════════════════════════════════════════════════════");

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./models/brain.sllm"));

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Build curriculum
    let curriculum = Curriculum::from_data_dir(&args.data);
    let phase_names: Vec<&str> = curriculum.phase_names().into_iter().collect();

    // Initialize progress tracker
    let progress_dir = output_path.parent().unwrap_or(Path::new("."));
    let mut progress = ProgressWriter::new(progress_dir, &args.name, &phase_names);

    // Phase 0: Build tokenizer from sampled corpus
    info!("════ Phase 0: Building BPE Tokenizer ════");
    progress.progress_mut().status = TrainingStatus::Tokenizing;
    progress.flush()?;

    let (tokenizer, mut brain) = if let Some(ref resume_path) = args.resume {
        info!("Resuming from: {}", resume_path.display());
        let loaded = BrainReader::read_owned(resume_path)?;
        info!(
            "Loaded model '{}': {} associations, {} tokens",
            loaded.header.model_name,
            loaded.brain.total_associations(),
            loaded.brain.tokens_trained()
        );
        (loaded.tokenizer, loaded.brain)
    } else {
        info!("Sampling corpus for tokenizer training...");
        let samples = curriculum.collect_tokenizer_samples(args.tokenizer_sample_lines);

        info!(
            "Training BPE tokenizer (target vocab: {})...",
            args.vocab_size
        );
        let trainer = BpeTrainer::new(args.vocab_size);
        let tokenizer = trainer.train(samples.iter().map(|s| s.as_str()));
        info!(
            "Tokenizer built: {} tokens, {} merges",
            tokenizer.vocab().len(),
            tokenizer.num_merges()
        );

        let brain = NgramBrain::new(args.max_associations);
        (tokenizer, brain)
    };

    progress.update_model_stats(
        brain.total_associations(),
        brain.tokens_trained(),
        tokenizer.vocab().len(),
    );
    progress.flush()?;

    // Train each curriculum phase
    for (phase_idx, phase) in curriculum.phases.iter().enumerate() {
        if shutdown.load(Ordering::Relaxed) {
            info!("Shutdown requested — saving checkpoint");
            save_checkpoint(&output_path, &args.name, &tokenizer, &brain)?;
            progress.flush()?;
            return Ok(());
        }

        info!("════ Phase {}: {} ════", phase_idx + 1, phase.name);

        // Update progress
        progress.progress_mut().current_phase_index = phase_idx;
        progress.progress_mut().current_phase = phase.name.clone();
        progress.start_phase();

        // Set total files for this phase
        let files = curriculum::list_text_files(&phase.data_path);
        if let Some(pp) = progress.current_phase_mut() {
            pp.total_files = files.len() as u64;
            pp.max_epochs = phase.max_epochs;
        }
        progress.flush()?;

        if files.is_empty() {
            warn!("Phase '{}': no data files found, skipping", phase.name);
            progress.advance_phase();
            continue;
        }

        // Load test data for evaluation
        let test_text = evaluator::load_test_data(&phase.data_path);

        // Train for up to max_epochs
        let mut previous_perplexity: Option<f64> = None;

        for epoch in 0..phase.max_epochs {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            info!("  Epoch {}/{}", epoch + 1, phase.max_epochs);

            // Update progress
            if let Some(pp) = progress.current_phase_mut() {
                pp.epoch = epoch + 1;
                pp.files_processed = 0;
            }

            // Train on all files in this phase
            train_on_files(
                &files,
                &tokenizer,
                &mut brain,
                &mut progress,
                &output_path,
                &args.name,
                args.checkpoint_interval,
                shutdown,
            )?;

            // Update stats
            progress.update_model_stats(
                brain.total_associations(),
                brain.tokens_trained(),
                tokenizer.vocab().len(),
            );

            // Evaluate
            if !test_text.is_empty() {
                progress.progress_mut().status = TrainingStatus::Evaluating;
                if let Some(pp) = progress.current_phase_mut() {
                    pp.status = PhaseStatus::Evaluating;
                }
                progress.flush()?;

                let eval_result = evaluator::evaluate_phase(
                    &brain,
                    &tokenizer,
                    &phase.name,
                    &test_text,
                    previous_perplexity,
                );

                // Record metrics
                if let Some(pp) = progress.current_phase_mut() {
                    pp.perplexity_history.push(eval_result.perplexity);
                    pp.coverage_history.push(eval_result.coverage);
                    pp.latest_perplexity = Some(eval_result.perplexity);
                    pp.latest_coverage = Some(eval_result.coverage);
                    if !eval_result.sample_generation.is_empty() {
                        pp.sample_generation = Some(eval_result.sample_generation);
                    }
                }

                previous_perplexity = Some(eval_result.perplexity);

                // Check convergence
                if eval_result.converged && epoch >= 1 {
                    info!(
                        "  ✓ Phase '{}' converged at epoch {}",
                        phase.name,
                        epoch + 1
                    );
                    break;
                }
            }

            // Checkpoint after each epoch
            info!("  Checkpointing...");
            save_checkpoint(&output_path, &args.name, &tokenizer, &brain)?;
            progress.progress_mut().status = TrainingStatus::Training;
            progress.flush()?;
        }

        // Consolidation: prune if configured
        if phase.prune_threshold > 0 {
            info!("  Consolidating (prune threshold={})...", phase.prune_threshold);
            progress.progress_mut().status = TrainingStatus::Consolidating;
            progress.flush()?;

            let before = brain.total_associations();
            brain.prune_all(phase.prune_threshold);
            let after = brain.total_associations();
            info!(
                "  Pruned {} associations ({} → {})",
                before - after,
                before,
                after
            );
        }

        // Mark phase complete
        progress.advance_phase();
        progress.update_model_stats(
            brain.total_associations(),
            brain.tokens_trained(),
            tokenizer.vocab().len(),
        );
        progress.flush()?;

        // Save after each phase
        save_checkpoint(&output_path, &args.name, &tokenizer, &brain)?;
    }

    // Final save
    info!("════ Training Complete ════");
    info!(
        "Model '{}': {} associations, {} tokens trained",
        args.name,
        brain.total_associations(),
        brain.tokens_trained()
    );
    progress.mark_converged();

    // Record final model size
    if let Ok(meta) = std::fs::metadata(&output_path) {
        progress.progress_mut().model.model_size_bytes = meta.len();
    }
    progress.flush()?;

    info!("Saved to: {}", output_path.display());
    info!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Run a single training pass (non-autonomous mode).
fn run_single_phase(args: &Args, shutdown: &Arc<AtomicBool>) -> Result<()> {
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./models/model.sllm"));

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (tokenizer, mut brain) = if let Some(ref resume_path) = args.resume {
        info!("Resuming from: {}", resume_path.display());
        let loaded =
            BrainReader::read_owned(resume_path).context("Failed to read existing model")?;
        info!(
            "Loaded model '{}': {} associations, {} tokens trained",
            loaded.header.model_name,
            loaded.brain.total_associations(),
            loaded.brain.tokens_trained()
        );
        (loaded.tokenizer, loaded.brain)
    } else {
        info!("Creating new model '{}'", args.name);
        info!(
            "Building tokenizer from training data (vocab_size={})...",
            args.vocab_size
        );

        let text_iter = collect_text_files(&args.data)?;
        let trainer = BpeTrainer::new(args.vocab_size);
        let tokenizer = trainer.train(text_iter.iter().map(|s| s.as_str()));
        info!(
            "Tokenizer built: {} tokens, {} merges",
            tokenizer.vocab().len(),
            tokenizer.num_merges()
        );

        let brain = NgramBrain::new(args.max_associations);
        (tokenizer, brain)
    };

    let files = curriculum::list_text_files(&args.data);
    info!("Found {} training files", files.len());

    let progress_dir = output_path.parent().unwrap_or(Path::new("."));
    let mut progress = ProgressWriter::new(progress_dir, &args.name, &["Single Phase"]);
    progress.start_phase();
    if let Some(pp) = progress.current_phase_mut() {
        pp.total_files = files.len() as u64;
    }
    progress.flush()?;

    train_on_files(
        &files,
        &tokenizer,
        &mut brain,
        &mut progress,
        &output_path,
        &args.name,
        args.checkpoint_interval,
        shutdown,
    )?;

    // Consolidation
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
    progress.mark_converged();
    progress.flush()?;

    info!(
        "Training complete: {} associations, {} tokens",
        brain.total_associations(),
        brain.tokens_trained(),
    );

    Ok(())
}

/// Train the brain on a list of files, updating progress.
fn train_on_files(
    files: &[PathBuf],
    tokenizer: &BpeTokenizer,
    brain: &mut NgramBrain,
    progress: &mut ProgressWriter,
    output_path: &Path,
    model_name: &str,
    checkpoint_interval: usize,
    shutdown: &Arc<AtomicBool>,
) -> Result<()> {
    let start_time = std::time::Instant::now();
    let mut files_this_pass = 0u64;
    let mut lines_this_pass = 0u64;

    for (i, file_path) in files.iter().enumerate() {
        if shutdown.load(Ordering::Relaxed) {
            info!("Shutdown requested — checkpointing...");
            save_checkpoint(output_path, model_name, tokenizer, brain)?;
            return Ok(());
        }

        // Stream file line-by-line to avoid loading huge files
        let file = match std::fs::File::open(file_path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Skipping {}: {}", file_path.display(), e);
                continue;
            }
        };

        let reader = std::io::BufReader::new(file);
        let mut file_tokens = Vec::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let tokens = tokenizer.encode(trimmed);
            file_tokens.extend_from_slice(&tokens);
            lines_this_pass += 1;

            // Train in chunks to avoid massive single sequences
            if file_tokens.len() >= 512 {
                brain.train_sequence(&file_tokens);
                file_tokens.clear();
            }
        }

        // Train remaining tokens
        if file_tokens.len() >= 5 {
            brain.train_sequence(&file_tokens);
        }

        files_this_pass += 1;

        // Update progress
        if let Some(pp) = progress.current_phase_mut() {
            pp.files_processed = files_this_pass;
            pp.lines_processed = lines_this_pass;
            pp.tokens_trained = brain.tokens_trained();
        }

        // Progress logging
        if files_this_pass % 50 == 0 || i == files.len() - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = files_this_pass as f64 / elapsed;
            let eta_secs = if rate > 0.0 {
                (files.len() as f64 - files_this_pass as f64) / rate
            } else {
                0.0
            };
            info!(
                "  [{}/{}] {} assoc | {} tokens | {:.1} files/s | ETA {:.0}s",
                files_this_pass,
                files.len(),
                brain.total_associations(),
                brain.tokens_trained(),
                rate,
                eta_secs,
            );
            progress.flush()?;
        }

        // Checkpoint
        if checkpoint_interval > 0 && files_this_pass as usize % checkpoint_interval == 0 {
            info!("  Checkpointing...");
            save_checkpoint(output_path, model_name, tokenizer, brain)?;
        }
    }

    Ok(())
}

/// Save a checkpoint of the current model.
fn save_checkpoint(
    output_path: &Path,
    model_name: &str,
    tokenizer: &BpeTokenizer,
    brain: &NgramBrain,
) -> Result<()> {
    BrainWriter::write(output_path, model_name, tokenizer, brain)?;
    Ok(())
}

/// Collect all text from files for tokenizer training (single-phase mode).
fn collect_text_files(path: &Path) -> Result<Vec<String>> {
    let mut texts = Vec::new();

    if path.is_file() {
        let text = std::fs::read_to_string(path)?;
        texts.push(text);
    } else if path.is_dir() {
        for entry in curriculum::list_text_files(path) {
            if let Ok(text) = std::fs::read_to_string(&entry) {
                // Limit per-file text for tokenizer training
                if text.len() < 1_000_000 {
                    texts.push(text);
                }
            }
        }
    }

    Ok(texts)
}
