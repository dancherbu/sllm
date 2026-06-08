//! RAG snippet storage trait and in-memory implementation.

/// A code/text snippet stored in the RAG system.
#[derive(Debug, Clone)]
pub struct Snippet {
    /// Unique identifier
    pub id: String,
    /// The text content
    pub content: String,
    /// Source file path (if applicable)
    pub source: Option<String>,
    /// Language (e.g., "python", "rust", "javascript")
    pub language: Option<String>,
}

/// Trait for persistent snippet storage.
///
/// Implemented by SQLite (Phase 4) and an in-memory mock (for testing).
pub trait SnippetStore: Send + Sync {
    /// Store a snippet. Overwrites if the ID already exists.
    fn put(&mut self, snippet: Snippet);

    /// Retrieve a snippet by ID.
    fn get(&self, id: &str) -> Option<&Snippet>;

    /// Retrieve multiple snippets by their IDs.
    fn get_many(&self, ids: &[String]) -> Vec<&Snippet>;

    /// Remove a snippet by ID.
    fn remove(&mut self, id: &str);

    /// Number of stored snippets.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory snippet store for testing.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    snippets: Vec<Snippet>,
}

impl SnippetStore for InMemoryStore {
    fn put(&mut self, snippet: Snippet) {
        self.snippets.retain(|s| s.id != snippet.id);
        self.snippets.push(snippet);
    }

    fn get(&self, id: &str) -> Option<&Snippet> {
        self.snippets.iter().find(|s| s.id == id)
    }

    fn get_many(&self, ids: &[String]) -> Vec<&Snippet> {
        self.snippets
            .iter()
            .filter(|s| ids.contains(&s.id))
            .collect()
    }

    fn remove(&mut self, id: &str) {
        self.snippets.retain(|s| s.id != id);
    }

    fn len(&self) -> usize {
        self.snippets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_store() {
        let mut store = InMemoryStore::default();

        store.put(Snippet {
            id: "1".to_string(),
            content: "def hello(): pass".to_string(),
            source: Some("main.py".to_string()),
            language: Some("python".to_string()),
        });

        assert_eq!(store.len(), 1);
        assert_eq!(store.get("1").unwrap().content, "def hello(): pass");

        store.remove("1");
        assert!(store.is_empty());
    }
}
