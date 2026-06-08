//! Token sampling strategies for text generation.
//!
//! Given a probability distribution over next tokens, select which
//! token to emit. Supports greedy, top-k, top-p, temperature scaling,
//! and repetition penalty.

use rand::Rng;

/// Token sampler with configurable strategies.
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Temperature for probability scaling (1.0 = neutral, <1.0 = sharper, >1.0 = flatter)
    pub temperature: f64,
    /// Top-k: only consider the k most likely tokens (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus): only consider tokens whose cumulative probability <= p (1.0 = disabled)
    pub top_p: f64,
    /// Repetition penalty multiplier (1.0 = no penalty, >1.0 = penalize repeats)
    pub repetition_penalty: f64,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
        }
    }
}

impl Sampler {
    /// Create a greedy sampler (always picks the highest probability token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }

    /// Sample from a probability distribution.
    ///
    /// `probs` is a slice of (token_id, probability) pairs, sorted by probability descending.
    /// `recent_tokens` is used for repetition penalty.
    ///
    /// Returns the selected token ID, or None if `probs` is empty.
    pub fn sample(
        &self,
        probs: &[(u16, f64)],
        recent_tokens: &[u16],
        rng: &mut impl Rng,
    ) -> Option<u16> {
        if probs.is_empty() {
            return None;
        }

        // Greedy: just pick the top
        if self.temperature == 0.0 || self.top_k == 1 {
            return Some(probs[0].0);
        }

        // Step 1: Apply repetition penalty
        let mut adjusted: Vec<(u16, f64)> = probs
            .iter()
            .map(|&(token, prob)| {
                if self.repetition_penalty != 1.0 && recent_tokens.contains(&token) {
                    (token, prob / self.repetition_penalty)
                } else {
                    (token, prob)
                }
            })
            .collect();

        // Step 2: Apply temperature
        if self.temperature != 1.0 {
            for item in &mut adjusted {
                // Scale log-probability by temperature, then re-exponentiate
                // For simplicity with raw probs: prob^(1/T) (sharper when T<1)
                item.1 = item.1.powf(1.0 / self.temperature);
            }
        }

        // Step 3: Apply top-k
        if self.top_k > 0 && adjusted.len() > self.top_k {
            adjusted.truncate(self.top_k);
        }

        // Re-sort after penalty/temperature adjustments
        adjusted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Step 4: Apply top-p (nucleus sampling)
        if self.top_p < 1.0 {
            let total: f64 = adjusted.iter().map(|x| x.1).sum();
            let mut cumulative = 0.0;
            let mut cutoff = adjusted.len();
            for (i, &(_, prob)) in adjusted.iter().enumerate() {
                cumulative += prob / total;
                if cumulative >= self.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            adjusted.truncate(cutoff);
        }

        // Step 5: Normalize and sample
        let total: f64 = adjusted.iter().map(|x| x.1).sum();
        if total <= 0.0 {
            return Some(adjusted[0].0);
        }

        let r: f64 = rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for &(token, prob) in &adjusted {
            cumulative += prob;
            if cumulative >= r {
                return Some(token);
            }
        }

        // Fallback to last token
        Some(adjusted.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_greedy_picks_highest() {
        let sampler = Sampler::greedy();
        let probs = vec![(10, 0.5), (20, 0.3), (30, 0.2)];
        let mut rng = make_rng();
        let result = sampler.sample(&probs, &[], &mut rng);
        assert_eq!(result, Some(10));
    }

    #[test]
    fn test_empty_probs() {
        let sampler = Sampler::default();
        let mut rng = make_rng();
        let result = sampler.sample(&[], &[], &mut rng);
        assert_eq!(result, None);
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = Sampler {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 100.0, // Very high penalty
        };
        // Token 10 has higher base prob but is penalized
        let probs = vec![(10, 0.5), (20, 0.49)];
        let recent = vec![10u16]; // Token 10 was recently generated
        let mut rng = make_rng();

        // With extreme penalty, greedy still picks the first but let's verify via non-greedy
        let sampler2 = Sampler {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 100.0,
        };
        // Token 10: 0.5/100 = 0.005, Token 20: 0.49
        // Token 20 should dominate
        let mut count_20 = 0;
        for _ in 0..100 {
            let r = sampler2.sample(&probs, &recent, &mut rng);
            if r == Some(20) {
                count_20 += 1;
            }
        }
        assert!(count_20 > 90, "Token 20 should dominate with high penalty");
    }

    #[test]
    fn test_sampling_distribution() {
        let sampler = Sampler {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let probs = vec![(10, 0.7), (20, 0.2), (30, 0.1)];
        let mut rng = make_rng();

        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            match sampler.sample(&probs, &[], &mut rng) {
                Some(10) => counts[0] += 1,
                Some(20) => counts[1] += 1,
                Some(30) => counts[2] += 1,
                _ => {}
            }
        }

        // Token 10 should appear most often
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }
}
