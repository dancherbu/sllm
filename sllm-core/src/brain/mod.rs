//! # Brain Module
//!
//! The associative "brain" of the sLLM. Uses Hebbian-style conditional counting
//! instead of gradient descent. No calculus, no floating-point weights.
//!
//! The brain maintains n-gram count tables (2-gram through 5-gram) and uses
//! interpolated smoothing to combine predictions across orders.

mod count_min;
mod count_table;
mod sampler;

pub use count_min::CountMinSketch;
pub use count_table::{CountTable, NgramBrain, BrainError};
pub use sampler::Sampler;
