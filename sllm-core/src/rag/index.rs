//! RAG text search index trait and in-memory implementation.

/// A scored search result from the RAG index.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique snippet ID
    pub id: String,
    /// Relevance score (higher = more relevant)
    pub score: f64,
}

/// Trait for text search indexes.
///
/// Implemented by BM25/tantivy (Phase 4) and an in-memory mock (for testing).
pub trait RagIndex: Send + Sync {
    /// Index a document with the given ID and text content.
    fn index(&mut self, id: &str, text: &str);

    /// Search for documents matching the query. Returns top-k results.
    fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult>;

    /// Remove a document from the index.
    fn remove(&mut self, id: &str);

    /// Number of documents in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory index for testing and small-scale use.
///
/// Uses naive substring matching. Replace with BM25/tantivy in Phase 4.
#[derive(Debug, Default)]
pub struct InMemoryIndex {
    documents: Vec<(String, String)>, // (id, text)
}

impl RagIndex for InMemoryIndex {
    fn index(&mut self, id: &str, text: &str) {
        // Remove existing entry with same ID
        self.documents.retain(|(doc_id, _)| doc_id != id);
        self.documents.push((id.to_string(), text.to_string()));
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .filter_map(|(id, text)| {
                let text_lower = text.to_lowercase();
                let matching_terms = query_terms
                    .iter()
                    .filter(|term| text_lower.contains(**term))
                    .count();

                if matching_terms > 0 {
                    let score = matching_terms as f64 / query_terms.len().max(1) as f64;
                    Some(SearchResult {
                        id: id.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    fn remove(&mut self, id: &str) {
        self.documents.retain(|(doc_id, _)| doc_id != id);
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_index() {
        let mut index = InMemoryIndex::default();
        index.index("1", "def fibonacci(n): return n");
        index.index("2", "fn main() { println!(\"hello\"); }");
        index.index("3", "function fibonacci(n) { return n; }");

        let results = index.search("fibonacci", 10);
        assert_eq!(results.len(), 2);

        let results = index.search("main println", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "2");
    }

    #[test]
    fn test_remove() {
        let mut index = InMemoryIndex::default();
        index.index("1", "hello world");
        assert_eq!(index.len(), 1);
        index.remove("1");
        assert_eq!(index.len(), 0);
    }
}
