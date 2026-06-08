//! Training progress tracker.
//!
//! Writes a JSON file (`training_progress.json`) that the monitor tool
//! can read to display live training status.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Complete training progress state, persisted to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    /// Current phase name
    pub current_phase: String,
    /// Current phase index (0-based)
    pub current_phase_index: usize,
    /// Total number of phases
    pub total_phases: usize,
    /// Overall status
    pub status: TrainingStatus,
    /// Per-phase progress
    pub phases: Vec<PhaseProgress>,
    /// Model stats
    pub model: ModelStats,
    /// Training start time (ISO 8601)
    pub started_at: String,
    /// Last update time (ISO 8601)
    pub updated_at: String,
    /// Elapsed seconds
    pub elapsed_secs: f64,
}

/// Training status enum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    /// Building tokenizer
    Tokenizing,
    /// Training on data
    Training,
    /// Running evaluation
    Evaluating,
    /// Consolidation/pruning pass
    Consolidating,
    /// Training complete — converged
    Converged,
    /// Training failed
    Failed,
}

/// Progress for a single training phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseProgress {
    /// Phase name (e.g., "Ashanti Twi", "English")
    pub name: String,
    /// Phase status
    pub status: PhaseStatus,
    /// Files processed in this phase
    pub files_processed: u64,
    /// Total files in this phase
    pub total_files: u64,
    /// Lines processed
    pub lines_processed: u64,
    /// Tokens trained in this phase
    pub tokens_trained: u64,
    /// Current epoch (for multi-epoch phases)
    pub epoch: u32,
    /// Max epochs for this phase
    pub max_epochs: u32,
    /// Perplexity history (one per epoch)
    pub perplexity_history: Vec<f64>,
    /// Coverage history (one per epoch)
    pub coverage_history: Vec<f64>,
    /// Latest perplexity
    pub latest_perplexity: Option<f64>,
    /// Latest coverage
    pub latest_coverage: Option<f64>,
    /// Sample generation from this phase
    pub sample_generation: Option<String>,
}

/// Phase completion status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    Pending,
    Active,
    Evaluating,
    Completed,
    Converged,
}

/// Current model statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Model name
    pub name: String,
    /// Total associations across all n-gram tables
    pub total_associations: u64,
    /// Total tokens trained
    pub total_tokens_trained: u64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model file size in bytes (last checkpoint)
    pub model_size_bytes: u64,
    /// Overall perplexity (weighted average)
    pub overall_perplexity: Option<f64>,
}

/// Writer that manages the progress file.
pub struct ProgressWriter {
    /// Path to the progress JSON file
    path: PathBuf,
    /// Current progress state
    progress: TrainingProgress,
    /// Training start instant (for elapsed time)
    start_instant: Instant,
}

impl ProgressWriter {
    /// Create a new progress writer.
    pub fn new(output_dir: &Path, model_name: &str, phase_names: &[&str]) -> Self {
        let path = output_dir.join("training_progress.json");

        let phases: Vec<PhaseProgress> = phase_names
            .iter()
            .map(|name| PhaseProgress {
                name: name.to_string(),
                status: PhaseStatus::Pending,
                files_processed: 0,
                total_files: 0,
                lines_processed: 0,
                tokens_trained: 0,
                epoch: 0,
                max_epochs: 1,
                perplexity_history: Vec::new(),
                coverage_history: Vec::new(),
                latest_perplexity: None,
                latest_coverage: None,
                sample_generation: None,
            })
            .collect();

        let now = chrono::Local::now().to_rfc3339();

        let progress = TrainingProgress {
            current_phase: phase_names.first().unwrap_or(&"unknown").to_string(),
            current_phase_index: 0,
            total_phases: phase_names.len(),
            status: TrainingStatus::Tokenizing,
            phases,
            model: ModelStats {
                name: model_name.to_string(),
                total_associations: 0,
                total_tokens_trained: 0,
                vocab_size: 0,
                model_size_bytes: 0,
                overall_perplexity: None,
            },
            started_at: now.clone(),
            updated_at: now,
            elapsed_secs: 0.0,
        };

        Self {
            path,
            progress,
            start_instant: Instant::now(),
        }
    }

    /// Get mutable reference to the current phase.
    pub fn current_phase_mut(&mut self) -> Option<&mut PhaseProgress> {
        let idx = self.progress.current_phase_index;
        self.progress.phases.get_mut(idx)
    }

    /// Get reference to the progress state.
    pub fn progress(&self) -> &TrainingProgress {
        &self.progress
    }

    /// Get mutable reference to the progress state.
    pub fn progress_mut(&mut self) -> &mut TrainingProgress {
        &mut self.progress
    }

    /// Advance to the next phase.
    pub fn advance_phase(&mut self) {
        if let Some(phase) = self.progress.phases.get_mut(self.progress.current_phase_index) {
            phase.status = PhaseStatus::Completed;
        }
        self.progress.current_phase_index += 1;
        if self.progress.current_phase_index < self.progress.total_phases {
            let name = self.progress.phases[self.progress.current_phase_index]
                .name
                .clone();
            self.progress.current_phase = name;
            self.progress.phases[self.progress.current_phase_index].status = PhaseStatus::Active;
        }
    }

    /// Mark current phase as active.
    pub fn start_phase(&mut self) {
        self.progress.status = TrainingStatus::Training;
        if let Some(phase) = self.current_phase_mut() {
            phase.status = PhaseStatus::Active;
        }
    }

    /// Update model stats.
    pub fn update_model_stats(
        &mut self,
        total_associations: u64,
        total_tokens: u64,
        vocab_size: usize,
    ) {
        self.progress.model.total_associations = total_associations;
        self.progress.model.total_tokens_trained = total_tokens;
        self.progress.model.vocab_size = vocab_size;
    }

    /// Mark training as converged.
    pub fn mark_converged(&mut self) {
        self.progress.status = TrainingStatus::Converged;
        if let Some(phase) = self.current_phase_mut() {
            phase.status = PhaseStatus::Converged;
        }
    }

    /// Write progress to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.progress.elapsed_secs = self.start_instant.elapsed().as_secs_f64();
        self.progress.updated_at = chrono::Local::now().to_rfc3339();

        let json = serde_json::to_string_pretty(&self.progress)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Atomic write: write to temp file then rename
        let tmp_path = self.path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &self.path)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_serialization() {
        let dir = std::env::temp_dir();
        let mut writer = ProgressWriter::new(&dir, "test-model", &["Phase A", "Phase B"]);

        writer.start_phase();
        if let Some(phase) = writer.current_phase_mut() {
            phase.files_processed = 42;
            phase.total_files = 100;
        }
        writer.update_model_stats(1000, 5000, 22000);
        writer.flush().unwrap();

        // Read back
        let path = dir.join("training_progress.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let progress: TrainingProgress = serde_json::from_str(&json).unwrap();

        assert_eq!(progress.current_phase, "Phase A");
        assert_eq!(progress.phases[0].files_processed, 42);
        assert_eq!(progress.model.total_associations, 1000);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_phase_advancement() {
        let dir = std::env::temp_dir();
        let mut writer = ProgressWriter::new(&dir, "test", &["A", "B", "C"]);

        writer.start_phase();
        assert_eq!(writer.progress().current_phase_index, 0);

        writer.advance_phase();
        assert_eq!(writer.progress().current_phase_index, 1);
        assert_eq!(writer.progress().current_phase, "B");
        assert_eq!(writer.progress().phases[0].status, PhaseStatus::Completed);
    }
}
