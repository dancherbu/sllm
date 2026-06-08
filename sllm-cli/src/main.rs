//! sLLM CLI — Unified command-line interface
//!
//! Usage:
//!   sllm train --data ./data/ --output ./models/model.sllm
//!   sllm run my-model
//!   sllm serve
//!   sllm list
//!   sllm info my-model
//!   sllm merge model-a.sllm model-b.sllm --output merged.sllm

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use rand::SeedableRng;

use sllm_core::brain::{NgramBrain, Sampler};
use sllm_core::format::{BrainReader, BrainWriter};
use sllm_core::tokenizer::{BpeTokenizer, BpeTrainer};

/// sLLM — Shallow Large Language Model
#[derive(Parser, Debug)]
#[command(name = "sllm", about = "A gradient-free, CPU-only language model", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start an interactive chat session with a model
    Run {
        /// Model name or path to .sllm file
        model: String,

        /// Models directory (for name lookup)
        #[arg(long, default_value = "./models")]
        models_dir: PathBuf,
    },

    /// Train a new model or continue training
    Train {
        /// Path to training data (file or directory)
        #[arg(short, long)]
        data: PathBuf,

        /// Output path for the trained model
        #[arg(short, long, default_value = "./models/model.sllm")]
        output: PathBuf,

        /// Model name
        #[arg(short, long, default_value = "sllm-base")]
        name: String,

        /// Resume from existing model
        #[arg(short, long)]
        resume: Option<PathBuf>,

        /// Target vocabulary size
        #[arg(long, default_value = "16384")]
        vocab_size: usize,
    },

    /// Start the inference API server
    Serve {
        /// Models directory
        #[arg(long, default_value = "./models")]
        models_dir: PathBuf,

        /// Port
        #[arg(short, long, default_value_t = sllm_core::DEFAULT_PORT)]
        port: u16,
    },

    /// List available models
    List {
        /// Models directory
        #[arg(long, default_value = "./models")]
        models_dir: PathBuf,
    },

    /// Show detailed model information
    Info {
        /// Path to .sllm file
        model: PathBuf,
    },

    /// Merge two models into one
    Merge {
        /// First model
        model_a: PathBuf,

        /// Second model
        model_b: PathBuf,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sllm=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { model, models_dir } => cmd_run(&model, &models_dir),
        Commands::Train {
            data,
            output,
            name,
            resume,
            vocab_size,
        } => cmd_train(&data, &output, &name, resume.as_deref(), vocab_size),
        Commands::Serve { models_dir, port } => {
            println!("Starting sLLM server on port {}...", port);
            println!("Run `sllm-run --models-dir {} --port {}` directly for the full server.", models_dir.display(), port);
            Ok(())
        }
        Commands::List { models_dir } => cmd_list(&models_dir),
        Commands::Info { model } => cmd_info(&model),
        Commands::Merge {
            model_a,
            model_b,
            output,
        } => cmd_merge(&model_a, &model_b, &output),
    }
}

fn cmd_run(model: &str, models_dir: &PathBuf) -> Result<()> {
    // Resolve model path
    let model_path = if PathBuf::from(model).exists() {
        PathBuf::from(model)
    } else {
        models_dir.join(format!("{}.sllm", model))
    };

    println!("Loading model from {}...", model_path.display());
    let loaded = BrainReader::read_owned(&model_path)
        .context("Failed to load model")?;

    println!(
        "Model '{}' loaded: {} associations, {} vocab tokens",
        loaded.header.model_name,
        loaded.brain.total_associations(),
        loaded.header.vocab_size,
    );
    println!("Type your prompt and press Enter. Type 'quit' to exit.\n");

    let sampler = Sampler::default();
    let mut rng = rand::rngs::StdRng::from_os_rng();

    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Encode and generate
        let mut tokens = loaded.tokenizer.encode(input);
        let max_tokens = 128;

        for _ in 0..max_tokens {
            let context = if tokens.len() > 4 {
                &tokens[tokens.len() - 4..]
            } else {
                &tokens
            };

            let predictions = loaded.brain.predict_next(context);
            if predictions.is_empty() {
                break;
            }

            let recent = if tokens.len() > 20 {
                &tokens[tokens.len() - 20..]
            } else {
                &tokens
            };

            match sampler.sample(&predictions, recent, &mut rng) {
                Some(token) => {
                    if token == sllm_core::tokenizer::SpecialToken::Eos.id() {
                        break;
                    }
                    // Print token immediately (streaming effect)
                    if let Some(s) = loaded.tokenizer.vocab().id_to_token(token) {
                        print!("{}", s);
                        io::stdout().flush()?;
                    }
                    tokens.push(token);
                }
                None => break,
            }
        }
        println!("\n");
    }

    Ok(())
}

fn cmd_train(
    data: &PathBuf,
    output: &PathBuf,
    name: &str,
    resume: Option<&std::path::Path>,
    vocab_size: usize,
) -> Result<()> {
    println!("Training not directly available via `sllm train`.");
    println!("Use `sllm-train` binary for full training capabilities:");
    println!(
        "  sllm-train --data {} --output {} --name {} --vocab-size {}",
        data.display(),
        output.display(),
        name,
        vocab_size,
    );
    if let Some(r) = resume {
        println!("  --resume {}", r.display());
    }
    Ok(())
}

fn cmd_list(models_dir: &PathBuf) -> Result<()> {
    if !models_dir.is_dir() {
        println!("Models directory not found: {}", models_dir.display());
        return Ok(());
    }

    let entries = std::fs::read_dir(models_dir)?;
    let mut found = false;

    println!("{:<20} {:>10} {:>15} {:>12}", "NAME", "VOCAB", "ASSOCIATIONS", "TOKENS");
    println!("{}", "-".repeat(60));

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("sllm") {
            match BrainReader::read_owned(&path) {
                Ok(loaded) => {
                    println!(
                        "{:<20} {:>10} {:>15} {:>12}",
                        loaded.header.model_name,
                        loaded.header.vocab_size,
                        loaded.header.total_associations,
                        loaded.header.training_tokens_seen,
                    );
                    found = true;
                }
                Err(e) => {
                    eprintln!("Error reading {}: {}", path.display(), e);
                }
            }
        }
    }

    if !found {
        println!("No models found in {}", models_dir.display());
    }

    Ok(())
}

fn cmd_info(model_path: &PathBuf) -> Result<()> {
    let loaded = BrainReader::read_owned(model_path)
        .context("Failed to read model")?;

    let h = &loaded.header;
    println!("=== sLLM Model Info ===");
    println!("Name:              {}", h.model_name);
    println!("Version:           {}", h.version);
    println!("Vocab size:        {}", h.vocab_size);
    println!("Context window:    {}", h.context_window);
    println!("N-gram orders:     2..{}", h.ngram_orders);
    println!("Associations:      {}", h.total_associations);
    println!("Tokens trained:    {}", h.training_tokens_seen);
    println!("Checksum:          {:#010x}", h.data_checksum);

    let file_size = std::fs::metadata(model_path)?.len();
    println!("File size:         {:.2} MB", file_size as f64 / 1_048_576.0);

    println!("\nBrain tables:");
    for table in loaded.brain.tables() {
        println!(
            "  {}-gram: {} contexts, {} associations",
            table.order(),
            table.num_contexts(),
            table.total_entries(),
        );
    }

    Ok(())
}

fn cmd_merge(model_a: &PathBuf, model_b: &PathBuf, _output: &PathBuf) -> Result<()> {
    println!("Loading model A from {}...", model_a.display());
    let _a = BrainReader::read_owned(model_a)?;

    println!("Loading model B from {}...", model_b.display());
    let _b = BrainReader::read_owned(model_b)?;

    // For v1, we just re-train model B's data onto model A's brain
    // True merging (additive counts) will be implemented in a future phase
    println!("Merge is not yet fully implemented (requires additive count combination).");
    println!("For now, train incrementally: sllm-train --resume model_a.sllm --data <new-data>");

    Ok(())
}
