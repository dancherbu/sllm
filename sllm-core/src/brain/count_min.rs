//! Count-Min Sketch — a probabilistic data structure for approximate counting.
//!
//! Used for memory-efficient approximate counting of lower-order n-grams
//! where exact counts aren't critical. Trades accuracy for space.
//!
//! A CMS uses `depth` independent hash functions, each mapping to a row of
//! `width` counters. To increment, hash the key with each function and
//! increment the corresponding counter. To query, take the minimum across
//! all rows (hence "Count-Min").

use std::hash::{Hash, Hasher};

/// A Count-Min Sketch for approximate frequency counting.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    /// 2D array of counters: [depth][width]
    counters: Vec<Vec<u32>>,
    /// Number of hash functions (rows)
    depth: usize,
    /// Number of buckets per row
    width: usize,
    /// Seeds for each hash function
    seeds: Vec<u64>,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch.
    ///
    /// - `width`: number of buckets per hash function (higher = more accurate)
    /// - `depth`: number of hash functions (higher = more accurate, but slower)
    ///
    /// Memory usage: `width * depth * 4` bytes.
    ///
    /// Recommended defaults: width=1_048_576 (2^20), depth=4 → ~16 MB.
    pub fn new(width: usize, depth: usize) -> Self {
        let counters = vec![vec![0u32; width]; depth];
        // Deterministic seeds for reproducibility
        let seeds: Vec<u64> = (0..depth).map(|i| 0x517cc1b727220a95_u64.wrapping_mul(i as u64 + 1)).collect();

        Self {
            counters,
            depth,
            width,
            seeds,
        }
    }

    /// Create a sketch with default parameters (~16 MB memory).
    pub fn default_size() -> Self {
        Self::new(1 << 20, 4)
    }

    /// Increment the count for a key.
    pub fn increment<K: Hash>(&mut self, key: &K) {
        for i in 0..self.depth {
            let idx = self.hash_index(key, i);
            self.counters[i][idx] = self.counters[i][idx].saturating_add(1);
        }
    }

    /// Estimate the count for a key (returns the minimum across all rows).
    pub fn estimate<K: Hash>(&self, key: &K) -> u32 {
        let mut min = u32::MAX;
        for i in 0..self.depth {
            let idx = self.hash_index(key, i);
            min = min.min(self.counters[i][idx]);
        }
        min
    }

    /// Increment by a specific amount.
    pub fn increment_by<K: Hash>(&mut self, key: &K, amount: u32) {
        for i in 0..self.depth {
            let idx = self.hash_index(key, i);
            self.counters[i][idx] = self.counters[i][idx].saturating_add(amount);
        }
    }

    /// Reset all counters to zero.
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
    }

    /// Apply a global decay: multiply all counters by `factor` (0.0..1.0).
    /// Useful for periodic consolidation ("sleep" pass).
    pub fn decay(&mut self, factor: f64) {
        for row in &mut self.counters {
            for counter in row.iter_mut() {
                *counter = (*counter as f64 * factor) as u32;
            }
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.width * self.depth * std::mem::size_of::<u32>()
    }

    /// Compute the bucket index for a key using hash function `i`.
    fn hash_index<K: Hash>(&self, key: &K, i: usize) -> usize {
        let mut hasher = SipLikeHasher::new(self.seeds[i]);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
}

/// A simple hash function seeded with a u64.
/// Uses FNV-1a-like mixing for speed.
struct SipLikeHasher {
    state: u64,
}

impl SipLikeHasher {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Hasher for SipLikeHasher {
    fn finish(&self) -> u64 {
        // Final avalanche mix
        let mut h = self.state;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_increment_and_estimate() {
        let mut cms = CountMinSketch::new(1024, 4);
        cms.increment(&"hello");
        cms.increment(&"hello");
        cms.increment(&"hello");
        cms.increment(&"world");

        assert!(cms.estimate(&"hello") >= 3);
        assert!(cms.estimate(&"world") >= 1);
        assert_eq!(cms.estimate(&"nonexistent"), 0);
    }

    #[test]
    fn test_increment_by() {
        let mut cms = CountMinSketch::new(1024, 4);
        cms.increment_by(&42u64, 100);
        assert!(cms.estimate(&42u64) >= 100);
    }

    #[test]
    fn test_clear() {
        let mut cms = CountMinSketch::new(256, 3);
        cms.increment(&"test");
        assert!(cms.estimate(&"test") > 0);
        cms.clear();
        assert_eq!(cms.estimate(&"test"), 0);
    }

    #[test]
    fn test_decay() {
        let mut cms = CountMinSketch::new(256, 3);
        for _ in 0..100 {
            cms.increment(&"frequent");
        }
        let before = cms.estimate(&"frequent");
        cms.decay(0.5);
        let after = cms.estimate(&"frequent");
        assert!(after < before);
        assert!(after >= 40); // ~50 after 0.5 decay
    }

    #[test]
    fn test_memory_size() {
        let cms = CountMinSketch::new(1 << 20, 4);
        // 2^20 * 4 * 4 bytes = 16 MB
        assert_eq!(cms.memory_bytes(), 4 * (1 << 20) * 4);
    }

    #[test]
    fn test_token_pair_counting() {
        // Test with actual token ID pairs (as used in sLLM)
        let mut cms = CountMinSketch::new(4096, 4);
        let pair: (u16, u16) = (100, 200);
        cms.increment(&pair);
        cms.increment(&pair);
        assert!(cms.estimate(&pair) >= 2);
    }
}
