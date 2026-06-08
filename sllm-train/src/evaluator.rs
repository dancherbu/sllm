//! Self-evaluation and convergence detection.
//!
//! After each training phase, runs evaluation benchmarks and
//! determines whether the model has converged (should stop training).

use sllm_core::brain::NgramBrain;
use sllm_core::eval;
use sllm_core::tokenizer::BpeTokenizer;
use std::path::Path;
use tracing::{info, warn};

/// Evaluation prompts for each language domain.
const TWI_PROMPTS: &[&str] = &[
    "Me din de ",
    "Ɛnnɛ yɛ ",
    "Ghana yɛ ",
    "Wo ho te sɛn? Me ho ",
];

const ENGLISH_PROMPTS: &[&str] = &[
    "Once upon a time ",
    "The quick brown fox ",
    "In the beginning ",
    "She walked to the ",
];

const CODE_PROMPTS: &[&str] = &[
    "def fibonacci(",
    "function handle",
    "const result = ",
    "async fn main",
];

/// Result of a convergence check.
#[derive(Debug)]
pub struct ConvergenceResult {
    /// Whether the model has converged for this phase
    pub converged: bool,
    /// Perplexity value
    pub perplexity: f64,
    /// Coverage score (0.0 to 1.0)
    pub coverage: f64,
    /// Perplexity delta from previous check (negative = improving)
    pub perplexity_delta: Option<f64>,
    /// Sample generation
    pub sample_generation: String,
    /// Whether the generation passed sanity checks
    pub generation_sane: bool,
}

/// Run evaluation on a specific language domain.
///
/// - Computes perplexity on `test_text`
/// - Generates text from known prompts
/// - Checks convergence against previous perplexity
pub fn evaluate_phase(
    brain: &NgramBrain,
    tokenizer: &BpeTokenizer,
    phase_name: &str,
    test_text: &str,
    previous_perplexity: Option<f64>,
) -> ConvergenceResult {
    info!("Evaluating phase: {}", phase_name);

    // 1. Compute perplexity
    let perplexity = eval::perplexity(brain, tokenizer, test_text);
    info!("  Perplexity: {:.2}", perplexity);

    // 2. Compute coverage
    let coverage = eval::coverage_score(brain, tokenizer, test_text);
    info!("  Coverage: {:.1}%", coverage * 100.0);

    // 3. Generate from prompts
    let prompts = match phase_name.to_lowercase().as_str() {
        name if name.contains("twi") => TWI_PROMPTS,
        name if name.contains("english") => ENGLISH_PROMPTS,
        name if name.contains("code") || name.contains("personal") => CODE_PROMPTS,
        _ => ENGLISH_PROMPTS,
    };

    let mut best_generation = String::new();
    let mut any_sane = false;

    for prompt in prompts {
        let generated = eval::generate(brain, tokenizer, prompt, 30);
        let is_sane = eval::generation_is_sane(&generated);
        if is_sane && generated.len() > best_generation.len() {
            best_generation = format!("{}{}", prompt, generated);
            any_sane = true;
        }
        info!(
            "  Gen [{}]: \"{}{}\" {}",
            if is_sane { "✓" } else { "✗" },
            prompt,
            &generated[..generated.len().min(60)],
            if is_sane { "" } else { "(degenerate)" }
        );
    }

    // 4. Check convergence
    let perplexity_delta = previous_perplexity.map(|prev| {
        if prev.is_finite() && perplexity.is_finite() {
            (perplexity - prev) / prev // Relative change
        } else {
            -1.0 // Still improving from infinity
        }
    });

    // Converged if:
    // - Perplexity is finite (model isn't degenerate)
    // - Perplexity delta is < 1% improvement (|delta| < 0.01) for 2+ checks
    // - Generation passes sanity checks
    let converged = perplexity.is_finite()
        && perplexity_delta
            .map(|d| d.abs() < 0.01)
            .unwrap_or(false)
        && any_sane;

    if converged {
        info!(
            "  ✓ Phase '{}' CONVERGED (ppl={:.2}, delta={:.3}%)",
            phase_name,
            perplexity,
            perplexity_delta.unwrap_or(0.0) * 100.0
        );
    } else {
        info!(
            "  → Phase '{}' not yet converged (ppl={:.2}, delta={:.3}%)",
            phase_name,
            perplexity,
            perplexity_delta.unwrap_or(-100.0) * 100.0
        );
    }

    ConvergenceResult {
        converged,
        perplexity,
        coverage,
        perplexity_delta,
        sample_generation: best_generation,
        generation_sane: any_sane,
    }
}

/// Load held-out test data for a training phase.
///
/// Takes the last 5% of lines from each data file in the phase directory.
pub fn load_test_data(data_dir: &Path) -> String {
    let mut test_lines = Vec::new();

    let txt_files: Vec<_> = if data_dir.is_dir() {
        walkdir_txt(data_dir)
    } else if data_dir.is_file() {
        vec![data_dir.to_path_buf()]
    } else {
        Vec::new()
    };

    for file_path in &txt_files {
        match std::fs::read_to_string(file_path) {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();
                let test_count = (lines.len() / 20).max(10).min(500); // 5%, min 10, max 500
                let start = lines.len().saturating_sub(test_count);
                for line in &lines[start..] {
                    if !line.trim().is_empty() {
                        test_lines.push(line.to_string());
                    }
                }
            }
            Err(e) => {
                warn!("Could not read test data from {}: {}", file_path.display(), e);
            }
        }
    }

    info!(
        "Loaded {} test lines from {} files",
        test_lines.len(),
        txt_files.len()
    );

    test_lines.join("\n")
}

/// Recursively find .txt files in a directory.
fn walkdir_txt(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    fn walk(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    if !name.starts_with('.') {
                        walk(&path, files);
                    }
                } else if path.extension().is_some_and(|e| e == "txt") {
                    files.push(path);
                }
            }
        }
    }

    walk(dir, &mut files);
    files.sort();
    files
}
