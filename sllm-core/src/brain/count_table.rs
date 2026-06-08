//! N-gram conditional count tables — the core "brain" of the sLLM.
//!
//! Stores how often token B follows context [A], [A,B] follows [X,Y], etc.
//! Learning = incrementing integer counters. Inference = looking up the
//! highest-count next tokens given a context.

use std::collections::HashMap;

/// A single n-gram count table mapping context → {next_token → count}.
///
/// For a bigram table (order=2), the context is a single token.
/// For a trigram table (order=3), the context is two tokens.
/// And so on up to 5-gram.
#[derive(Debug, Clone)]
pub struct CountTable {
    /// The n-gram order (2 = bigram, 3 = trigram, etc.)
    order: u8,
    /// context (as token ID sequence) → { next_token_id → count }
    counts: HashMap<Vec<u16>, HashMap<u16, u32>>,
    /// Total number of (context, next_token) associations stored
    total_entries: u64,
    /// Maximum entries before eviction (0 = unlimited)
    max_entries: u64,
}

impl CountTable {
    /// Create a new count table for the given n-gram order.
    ///
    /// `order` must be >= 2 (bigram minimum).
    /// `max_entries` limits memory usage (0 = unlimited).
    pub fn new(order: u8, max_entries: u64) -> Self {
        assert!(order >= 2, "n-gram order must be at least 2");
        Self {
            order,
            counts: HashMap::new(),
            total_entries: 0,
            max_entries,
        }
    }

    /// The n-gram order of this table.
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Context length = order - 1 (number of preceding tokens needed).
    pub fn context_len(&self) -> usize {
        (self.order - 1) as usize
    }

    /// Record an observation: `next_token` followed `context`.
    ///
    /// `context` must have exactly `self.context_len()` elements.
    pub fn update(&mut self, context: &[u16], next_token: u16) {
        debug_assert_eq!(
            context.len(),
            self.context_len(),
            "context length mismatch: expected {}, got {}",
            self.context_len(),
            context.len()
        );

        let entry = self
            .counts
            .entry(context.to_vec())
            .or_insert_with(HashMap::new);

        let count = entry.entry(next_token).or_insert(0);
        if *count == 0 {
            self.total_entries += 1;
        }
        *count = count.saturating_add(1);

        // Evict if over budget
        if self.max_entries > 0 && self.total_entries > self.max_entries {
            self.evict_lowest();
        }
    }

    /// Predict the next token given a context.
    ///
    /// Returns a vec of (token_id, probability) sorted by probability descending.
    /// `context` must have exactly `self.context_len()` elements.
    pub fn predict(&self, context: &[u16]) -> Vec<(u16, f64)> {
        if context.len() != self.context_len() {
            return Vec::new();
        }

        let Some(next_counts) = self.counts.get(context) else {
            return Vec::new();
        };

        let total: u64 = next_counts.values().map(|&c| c as u64).sum();
        if total == 0 {
            return Vec::new();
        }

        let mut predictions: Vec<(u16, f64)> = next_counts
            .iter()
            .map(|(&token, &count)| (token, count as f64 / total as f64))
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        predictions
    }

    /// Prune all associations with count below `min_count`.
    pub fn prune(&mut self, min_count: u32) {
        let mut empty_contexts = Vec::new();

        for (context, next_counts) in &mut self.counts {
            let before = next_counts.len() as u64;
            next_counts.retain(|_, count| *count >= min_count);
            let after = next_counts.len() as u64;
            self.total_entries -= before - after;

            if next_counts.is_empty() {
                empty_contexts.push(context.clone());
            }
        }

        for ctx in empty_contexts {
            self.counts.remove(&ctx);
        }
    }

    /// Total number of unique (context, next_token) associations.
    pub fn total_entries(&self) -> u64 {
        self.total_entries
    }

    /// Number of unique contexts stored.
    pub fn num_contexts(&self) -> usize {
        self.counts.len()
    }

    /// Evict the lowest-count entries to stay within budget.
    fn evict_lowest(&mut self) {
        // Simple strategy: prune entries with count == 1
        self.prune(2);

        // If still over, prune count == 2, etc.
        let mut threshold = 3u32;
        while self.max_entries > 0 && self.total_entries > self.max_entries && threshold < 100 {
            self.prune(threshold);
            threshold += 1;
        }
    }

    /// Serialize this count table to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.push(self.order);
        buf.extend_from_slice(&(self.counts.len() as u64).to_le_bytes());

        for (context, next_counts) in &self.counts {
            // Context tokens
            for &token in context {
                buf.extend_from_slice(&token.to_le_bytes());
            }
            // Number of next tokens
            buf.extend_from_slice(&(next_counts.len() as u32).to_le_bytes());
            // Each (next_token, count) pair
            for (&token, &count) in next_counts {
                buf.extend_from_slice(&token.to_le_bytes());
                buf.extend_from_slice(&count.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize a count table from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), BrainError> {
        let mut pos = 0;

        if data.is_empty() {
            return Err(BrainError::TruncatedData);
        }

        let order = data[pos];
        pos += 1;

        if pos + 8 > data.len() {
            return Err(BrainError::TruncatedData);
        }
        let num_contexts = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let context_len = (order - 1) as usize;
        let mut table = CountTable::new(order, 0);

        for _ in 0..num_contexts {
            // Read context
            let mut context = Vec::with_capacity(context_len);
            for _ in 0..context_len {
                if pos + 2 > data.len() {
                    return Err(BrainError::TruncatedData);
                }
                let token = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                pos += 2;
                context.push(token);
            }

            // Read number of next tokens
            if pos + 4 > data.len() {
                return Err(BrainError::TruncatedData);
            }
            let num_next = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            // Read each (next_token, count)
            let mut next_counts = HashMap::with_capacity(num_next);
            for _ in 0..num_next {
                if pos + 6 > data.len() {
                    return Err(BrainError::TruncatedData);
                }
                let token = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                pos += 2;
                let count = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;

                next_counts.insert(token, count);
                table.total_entries += 1;
            }

            table.counts.insert(context, next_counts);
        }

        Ok((table, pos))
    }
}

/// The full n-gram brain holding multiple count tables.
///
/// Combines predictions from bigram through 5-gram tables using
/// interpolation weights (higher-order n-grams get more weight).
#[derive(Debug, Clone)]
pub struct NgramBrain {
    /// Count tables for each n-gram order (indices 0..4 → 2-gram..5-gram)
    tables: Vec<CountTable>,
    /// Interpolation weights for each order (must sum to ~1.0)
    weights: Vec<f64>,
    /// Total tokens seen during training
    tokens_trained: u64,
}

impl NgramBrain {
    /// Create a new brain with tables for 2-gram through 5-gram.
    ///
    /// `max_entries_per_table` limits memory per table (0 = unlimited).
    pub fn new(max_entries_per_table: u64) -> Self {
        let tables = (2..=5u8)
            .map(|order| CountTable::new(order, max_entries_per_table))
            .collect();

        // Default interpolation weights: higher order = higher weight
        // These are tuned for code generation where longer context matters.
        let weights = vec![0.05, 0.15, 0.30, 0.50]; // 2-gram, 3-gram, 4-gram, 5-gram

        Self {
            tables,
            weights,
            tokens_trained: 0,
        }
    }

    /// Train on a sequence of token IDs.
    ///
    /// Feeds sliding windows of tokens to each count table.
    pub fn train_sequence(&mut self, tokens: &[u16]) {
        for (i, table) in self.tables.iter_mut().enumerate() {
            let order = (i + 2) as usize; // 2, 3, 4, 5
            if tokens.len() < order {
                continue;
            }
            for window in tokens.windows(order) {
                let context = &window[..order - 1];
                let next_token = window[order - 1];
                table.update(context, next_token);
            }
        }
        self.tokens_trained += tokens.len() as u64;
    }

    /// Predict the next token given a context.
    ///
    /// Uses interpolated smoothing across all n-gram orders.
    /// `context` should be at least 4 tokens long for best results.
    pub fn predict_next(&self, context: &[u16]) -> Vec<(u16, f64)> {
        let mut combined: HashMap<u16, f64> = HashMap::new();

        for (i, table) in self.tables.iter().enumerate() {
            let ctx_len = table.context_len();
            if context.len() < ctx_len {
                continue;
            }
            let ctx = &context[context.len() - ctx_len..];
            let predictions = table.predict(ctx);
            let weight = self.weights[i];

            for (token, prob) in predictions {
                *combined.entry(token).or_insert(0.0) += weight * prob;
            }
        }

        let mut result: Vec<(u16, f64)> = combined.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    /// Total associations across all tables.
    pub fn total_associations(&self) -> u64 {
        self.tables.iter().map(|t| t.total_entries()).sum()
    }

    /// Total tokens seen during training.
    pub fn tokens_trained(&self) -> u64 {
        self.tokens_trained
    }

    /// Prune all tables, removing associations below `min_count`.
    pub fn prune_all(&mut self, min_count: u32) {
        for table in &mut self.tables {
            table.prune(min_count);
        }
    }

    /// Access the individual count tables.
    pub fn tables(&self) -> &[CountTable] {
        &self.tables
    }

    /// Serialize the entire brain to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Number of tables
        buf.extend_from_slice(&(self.tables.len() as u8).to_le_bytes());

        // Weights
        for &w in &self.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }

        // Tokens trained
        buf.extend_from_slice(&self.tokens_trained.to_le_bytes());

        // Each table
        for table in &self.tables {
            let table_bytes = table.to_bytes();
            buf.extend_from_slice(&(table_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(&table_bytes);
        }

        buf
    }

    /// Deserialize a brain from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), BrainError> {
        let mut pos = 0;

        if data.is_empty() {
            return Err(BrainError::TruncatedData);
        }
        let num_tables = data[pos] as usize;
        pos += 1;

        // Weights
        let mut weights = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            if pos + 8 > data.len() {
                return Err(BrainError::TruncatedData);
            }
            let w = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            weights.push(w);
        }

        // Tokens trained
        if pos + 8 > data.len() {
            return Err(BrainError::TruncatedData);
        }
        let tokens_trained = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Tables
        let mut tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            if pos + 8 > data.len() {
                return Err(BrainError::TruncatedData);
            }
            let table_size = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            if pos + table_size > data.len() {
                return Err(BrainError::TruncatedData);
            }
            let (table, _consumed) = CountTable::from_bytes(&data[pos..pos + table_size])?;
            pos += table_size;
            tables.push(table);
        }

        Ok((
            Self {
                tables,
                weights,
                tokens_trained,
            },
            pos,
        ))
    }
}

/// Errors from brain operations.
#[derive(Debug, thiserror::Error)]
pub enum BrainError {
    #[error("brain data is truncated")]
    TruncatedData,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_table_basic() {
        let mut table = CountTable::new(2, 0);
        table.update(&[10], 20);
        table.update(&[10], 20);
        table.update(&[10], 30);

        let preds = table.predict(&[10]);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].0, 20); // Higher count
        assert!((preds[0].1 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_count_table_empty_context() {
        let table = CountTable::new(2, 0);
        let preds = table.predict(&[999]);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_count_table_prune() {
        let mut table = CountTable::new(2, 0);
        table.update(&[10], 20); // count=1
        table.update(&[10], 30);
        table.update(&[10], 30); // count=2

        table.prune(2);
        let preds = table.predict(&[10]);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, 30);
    }

    #[test]
    fn test_ngram_brain_train_and_predict() {
        let mut brain = NgramBrain::new(0);
        // Sequence: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        brain.train_sequence(&tokens);

        assert!(brain.total_associations() > 0);
        assert_eq!(brain.tokens_trained(), 10);

        // After [1, 2, 3, 4], the brain should predict 5
        let context = vec![1, 2, 3, 4];
        let preds = brain.predict_next(&context);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 5);
    }

    #[test]
    fn test_count_table_serialization_roundtrip() {
        let mut table = CountTable::new(3, 0);
        table.update(&[1, 2], 3);
        table.update(&[1, 2], 3);
        table.update(&[1, 2], 4);
        table.update(&[5, 6], 7);

        let bytes = table.to_bytes();
        let (restored, consumed) = CountTable::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.order(), 3);
        assert_eq!(restored.total_entries(), table.total_entries());
    }

    #[test]
    fn test_ngram_brain_serialization_roundtrip() {
        let mut brain = NgramBrain::new(0);
        brain.train_sequence(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let bytes = brain.to_bytes();
        let (restored, _) = NgramBrain::from_bytes(&bytes).unwrap();
        assert_eq!(restored.tokens_trained(), brain.tokens_trained());
        assert_eq!(
            restored.total_associations(),
            brain.total_associations()
        );
    }
}
