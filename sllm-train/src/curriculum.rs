//! Phased curriculum manager.
//!
//! Defines the training curriculum (which data to train on, in what order)
//! and manages automatic phase transitions.

use std::path::{Path, PathBuf};
use tracing::info;

/// A single training phase definition.
#[derive(Debug, Clone)]
pub struct Phase {
    /// Human-readable phase name
    pub name: String,
    /// Path to training data (file or directory)
    pub data_path: PathBuf,
    /// Maximum epochs for this phase (0 = until converged)
    pub max_epochs: u32,
    /// Prune threshold after this phase (0 = no pruning)
    pub prune_threshold: u32,
    /// Whether this phase contributes to tokenizer training
    pub include_in_tokenizer: bool,
}

/// The full training curriculum.
#[derive(Debug)]
pub struct Curriculum {
    /// Ordered list of training phases
    pub phases: Vec<Phase>,
}

impl Curriculum {
    /// Build the default sLLM curriculum from the data directory.
    ///
    /// Expects the directory structure:
    /// ```text
    /// data/
    ///   twi/        → Phase 0: Ashanti Twi
    ///   english/    → Phase 1: English
    ///   code/       → Phase 2: Public Code
    ///   personal/   → Phase 3: Personal Code
    /// ```
    pub fn from_data_dir(data_dir: &Path) -> Self {
        let mut phases = Vec::new();

        // Phase 0: Ashanti Twi (mother tongue first)
        let twi_dir = data_dir.join("twi");
        if twi_dir.exists() {
            phases.push(Phase {
                name: "Ashanti Twi 🇬🇭".to_string(),
                data_path: twi_dir,
                max_epochs: 3,
                prune_threshold: 0,
                include_in_tokenizer: true,
            });
        }

        // Phase 1: English
        let english_dir = data_dir.join("english");
        if english_dir.exists() {
            phases.push(Phase {
                name: "English".to_string(),
                data_path: english_dir,
                max_epochs: 2,
                prune_threshold: 0,
                include_in_tokenizer: true,
            });
        }

        // Phase 2: Public Code
        let code_dir = data_dir.join("code");
        if code_dir.exists() {
            phases.push(Phase {
                name: "Public Code".to_string(),
                data_path: code_dir,
                max_epochs: 2,
                prune_threshold: 0,
                include_in_tokenizer: true,
            });
        }

        // Phase 3: Personal Code
        let personal_dir = data_dir.join("personal");
        if personal_dir.exists() {
            phases.push(Phase {
                name: "Personal Code".to_string(),
                data_path: personal_dir,
                max_epochs: 3,
                prune_threshold: 0,
                include_in_tokenizer: true,
            });
        }

        // Phase 4: Refinement (all data combined, multi-epoch)
        phases.push(Phase {
            name: "Refinement (all data)".to_string(),
            data_path: data_dir.to_path_buf(),
            max_epochs: 3, // Will stop early if converged
            prune_threshold: 2, // Prune singletons after refinement
            include_in_tokenizer: false,
        });

        info!("Curriculum: {} phases", phases.len());
        for (i, phase) in phases.iter().enumerate() {
            info!(
                "  Phase {}: {} (max {} epochs, data: {})",
                i,
                phase.name,
                phase.max_epochs,
                phase.data_path.display()
            );
        }

        Self { phases }
    }

    /// Get phase names for the progress tracker.
    pub fn phase_names(&self) -> Vec<&str> {
        self.phases.iter().map(|p| p.name.as_str()).collect()
    }

    /// Collect text files for tokenizer training (sampled from all phases).
    ///
    /// Samples up to `max_lines_per_phase` lines from each phase to keep
    /// memory reasonable during BPE training.
    pub fn collect_tokenizer_samples(
        &self,
        max_lines_per_phase: usize,
    ) -> Vec<String> {
        let mut samples = Vec::new();

        for phase in &self.phases {
            if !phase.include_in_tokenizer {
                continue;
            }

            let files = list_text_files(&phase.data_path);
            let mut phase_lines = Vec::new();

            for file_path in &files {
                match std::fs::read_to_string(file_path) {
                    Ok(content) => {
                        for line in content.lines() {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() && trimmed.len() > 3 {
                                phase_lines.push(trimmed.to_string());
                            }
                            if phase_lines.len() >= max_lines_per_phase {
                                break;
                            }
                        }
                    }
                    Err(_) => continue,
                }
                if phase_lines.len() >= max_lines_per_phase {
                    break;
                }
            }

            info!(
                "  Tokenizer sample: {} — {} lines",
                phase.name,
                phase_lines.len()
            );
            samples.extend(phase_lines);
        }

        info!("Total tokenizer samples: {} lines", samples.len());
        samples
    }
}

/// List all text/code files in a path recursively.
pub fn list_text_files(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
        return files;
    }

    if !path.is_dir() {
        return files;
    }

    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    if !name.starts_with('.')
                        && name != "node_modules"
                        && name != "target"
                        && name != "__pycache__"
                        && name != ".venv"
                    {
                        walk(&path, files);
                    }
                } else if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if matches!(
                            ext.as_str(),
                            "txt" | "md" | "py" | "rs" | "js" | "ts" | "jsx" | "tsx"
                                | "c" | "h" | "cpp" | "hpp" | "go" | "java" | "rb"
                                | "sh" | "bash" | "zsh" | "toml" | "yaml" | "yml"
                                | "json" | "html" | "css"
                        ) {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    walk(path, &mut files);
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_phase_names() {
        // Test with a temp directory structure
        let tmp = std::env::temp_dir().join("sllm_test_curriculum");
        let _ = std::fs::create_dir_all(tmp.join("twi"));
        let _ = std::fs::create_dir_all(tmp.join("english"));

        let curriculum = Curriculum::from_data_dir(&tmp);
        let names = curriculum.phase_names();
        assert!(names.len() >= 2); // twi + english + refinement
        assert!(names[0].contains("Twi"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
