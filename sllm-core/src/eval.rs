//! Evaluation metrics for n-gram language models.
//!
//! Provides perplexity computation, text generation, and coverage scoring
//! for self-checking training convergence.
//!
//! These are NOT gradient-based metrics — they measure the quality of
//! the conditional count tables directly.

use crate::brain::NgramBrain;
use crate::tokenizer::BpeTokenizer;

/// Compute perplexity of a trained model on test text.
///
/// Perplexity = exp(-1/N * Σ log P(token_i | context_i))
///
/// Lower perplexity = better model. For n-gram count tables,
/// P(token | context) is computed via interpolated lookups.
///
/// Returns `f64::INFINITY` if the model assigns zero probability
/// to too many tokens (degenerate model).
pub fn perplexity(brain: &NgramBrain, tokenizer: &BpeTokenizer, test_text: &str) -> f64 {
    let tokens = tokenizer.encode(test_text);
    if tokens.len() < 5 {
        return f64::INFINITY;
    }

    let mut log_prob_sum = 0.0f64;
    let mut count = 0u64;
    let mut zero_prob_count = 0u64;

    // Start from position 4 (need at least 4 tokens of context for 5-gram)
    for i in 4..tokens.len() {
        let context = &tokens[i.saturating_sub(4)..i];
        let predictions = brain.predict_next(context);

        let target = tokens[i];
        let prob = predictions
            .iter()
            .find(|(tok, _)| *tok == target)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        if prob > 0.0 {
            log_prob_sum += prob.ln();
        } else {
            // Smoothing: assign a tiny probability to unseen tokens
            // This prevents infinite perplexity from a single OOV token
            log_prob_sum += (1e-10_f64).ln();
            zero_prob_count += 1;
        }
        count += 1;
    }

    if count == 0 {
        return f64::INFINITY;
    }

    // If more than 50% of tokens have zero probability, model is degenerate
    if zero_prob_count as f64 / count as f64 > 0.5 {
        return f64::INFINITY;
    }

    let avg_neg_log_prob = -log_prob_sum / count as f64;
    avg_neg_log_prob.exp()
}

/// Generate text from a prompt using the trained model.
///
/// Uses greedy decoding (always picks the highest-probability next token)
/// for deterministic, reproducible evaluation.
///
/// Returns the generated text (excluding the prompt).
pub fn generate(
    brain: &NgramBrain,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    let prompt_len = tokens.len();

    if prompt_len < 2 {
        return String::new();
    }

    for _ in 0..max_tokens {
        // Use up to 4 tokens of context
        let ctx_start = tokens.len().saturating_sub(4);
        let context = &tokens[ctx_start..];
        let predictions = brain.predict_next(context);

        if predictions.is_empty() {
            break;
        }

        // Greedy: pick highest probability
        let next_token = predictions[0].0;

        // Stop on EOS-like tokens or repetition
        if next_token == 0 {
            break;
        }

        tokens.push(next_token);

        // Stop if generating repetitive output (same 4-token sequence repeating)
        if tokens.len() >= prompt_len + 8 {
            let last4 = &tokens[tokens.len() - 4..];
            let prev4 = &tokens[tokens.len() - 8..tokens.len() - 4];
            if last4 == prev4 {
                // Remove the repeated portion
                tokens.truncate(tokens.len() - 4);
                break;
            }
        }
    }

    // Decode only the generated portion
    tokenizer.decode(&tokens[prompt_len..])
}

/// Compute coverage score: what fraction of test tokens get a
/// non-zero probability prediction from the model.
///
/// Returns a value between 0.0 (model knows nothing) and 1.0
/// (model has seen patterns for every token in the test set).
pub fn coverage_score(brain: &NgramBrain, tokenizer: &BpeTokenizer, test_text: &str) -> f64 {
    let tokens = tokenizer.encode(test_text);
    if tokens.len() < 5 {
        return 0.0;
    }

    let mut covered = 0u64;
    let mut total = 0u64;

    for i in 4..tokens.len() {
        let context = &tokens[i.saturating_sub(4)..i];
        let predictions = brain.predict_next(context);

        let target = tokens[i];
        let has_prediction = predictions.iter().any(|(tok, _)| *tok == target);

        if has_prediction {
            covered += 1;
        }
        total += 1;
    }

    if total == 0 {
        return 0.0;
    }

    covered as f64 / total as f64
}

/// Result of a full evaluation run across multiple languages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalReport {
    /// Per-language evaluation results
    pub languages: Vec<LangEval>,
    /// Overall weighted perplexity
    pub overall_perplexity: f64,
    /// Overall coverage
    pub overall_coverage: f64,
    /// Sample generations
    pub generations: Vec<GenerationSample>,
}

/// Evaluation results for a single language.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LangEval {
    /// Language name (e.g., "twi", "english", "python")
    pub name: String,
    /// Perplexity on held-out test data
    pub perplexity: f64,
    /// Coverage score (0.0 to 1.0)
    pub coverage: f64,
    /// Number of test tokens evaluated
    pub test_tokens: u64,
}

/// A sample generation for sanity checking.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationSample {
    /// Language of the prompt
    pub language: String,
    /// The prompt given
    pub prompt: String,
    /// The generated continuation
    pub generated: String,
    /// Whether the generation passes sanity checks
    pub is_sane: bool,
}

/// Check if a generated text is "sane" (not degenerate).
///
/// A generation is sane if:
/// - It's not empty
/// - It's not all whitespace
/// - It doesn't repeat the same character/word excessively
/// - It contains at least some alphabetic characters
pub fn generation_is_sane(generated: &str) -> bool {
    let trimmed = generated.trim();

    // Must be non-empty
    if trimmed.is_empty() || trimmed.len() < 3 {
        return false;
    }

    // Must contain some alphabetic characters
    let alpha_ratio = trimmed.chars().filter(|c| c.is_alphabetic()).count() as f64
        / trimmed.len() as f64;
    if alpha_ratio < 0.1 {
        return false;
    }

    // Check for excessive single-character repetition
    if let Some(first_char) = trimmed.chars().next() {
        let same_char_count = trimmed.chars().filter(|&c| c == first_char).count();
        if same_char_count as f64 / trimmed.len() as f64 > 0.8 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::NgramBrain;
    use crate::tokenizer::BpeTrainer;

    fn make_trained_model() -> (NgramBrain, BpeTokenizer) {
        let texts = vec![
            "the cat sat on the mat",
            "the dog sat on the rug",
            "the cat ate the fish",
            "the dog ate the bone",
            "the cat sat on the mat again",
            "the dog sat on the rug again",
        ];
        let trainer = BpeTrainer::new(50);
        let tokenizer = trainer.train(texts.iter());

        let mut brain = NgramBrain::new(0);
        for text in &texts {
            let tokens = tokenizer.encode(text);
            brain.train_sequence(&tokens);
        }

        (brain, tokenizer)
    }

    #[test]
    fn test_perplexity_finite() {
        let (brain, tokenizer) = make_trained_model();
        let ppl = perplexity(&brain, &tokenizer, "the cat sat on the mat");
        assert!(ppl.is_finite());
        assert!(ppl > 0.0);
    }

    #[test]
    fn test_perplexity_lower_for_seen_data() {
        let (brain, tokenizer) = make_trained_model();
        let ppl_seen = perplexity(&brain, &tokenizer, "the cat sat on the mat");
        let ppl_unseen = perplexity(&brain, &tokenizer, "a bird flew over the house");
        // Seen data should have lower perplexity (better predictions)
        assert!(ppl_seen < ppl_unseen);
    }

    #[test]
    fn test_generate_non_empty() {
        let (brain, tokenizer) = make_trained_model();
        let output = generate(&brain, &tokenizer, "the cat", 20);
        // Should generate something (model has patterns for "the cat")
        assert!(!output.is_empty());
    }

    #[test]
    fn test_coverage_score_range() {
        let (brain, tokenizer) = make_trained_model();
        let score = coverage_score(&brain, &tokenizer, "the cat sat on the mat");
        assert!(score >= 0.0 && score <= 1.0);
        // Seen data should have decent coverage
        assert!(score > 0.3);
    }

    #[test]
    fn test_generation_sanity_checks() {
        assert!(generation_is_sane("the cat sat on the mat"));
        assert!(!generation_is_sane(""));
        assert!(!generation_is_sane("   "));
        assert!(!generation_is_sane("aaaaaaaaaa"));
        assert!(!generation_is_sane("...."));
    }
}
